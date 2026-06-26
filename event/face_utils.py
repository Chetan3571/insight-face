import logging
import os
import tempfile
import zipfile
from urllib.request import urlretrieve

import cv2
import gdown
import insightface
import numpy as np
import onnxruntime
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from insightface.utils import face_align as _face_align

logger = logging.getLogger('event')

MODEL_PACK = 'adaface'
FP32_ONNX = 'adaface_ir101_webface12m.onnx'          # canonical download name (always fetched)
RECOG_ONNX = os.environ.get('ADAFACE_MODEL_FILENAME', FP32_ONNX)   # active model (FP32 or INT8)
RECOG_GDRIVE_ID = '1dgMFOASKnaujQcCL4sSYkKOkBrmXUUU1'
# SCRFD detector + 106-point landmarks (InsightFace ONNX helpers for the AdaFace pipeline)
DET_ONNX = 'det_10g.onnx'
LANDMARK_ONNX = '2d106det.onnx'
AUX_ONNX_FILES = (DET_ONNX, LANDMARK_ONNX)
# One-time zip fetch for detector/landmark ONNX (extracted into the adaface pack only)
AUX_ONNX_ZIP_URL = (
    'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'
)

MIN_FACE_PX = 40


def _min_face_px() -> int:
    return int(os.environ.get('MIN_FACE_PX', str(MIN_FACE_PX)))


def _rec_batch_size() -> int:
    return int(os.environ.get('REC_BATCH_SIZE', '16'))


def _execution_provider_config(ctx_id: int) -> tuple[list[str], list[dict]]:
    available = onnxruntime.get_available_providers()
    if ctx_id >= 0:
        if 'CUDAExecutionProvider' not in available:
            raise RuntimeError(
                'INSIGHTFACE_CTX_ID is set to GPU, but ONNX Runtime CUDA is not available. '
                f'Available providers: {available}. Install onnxruntime-gpu in the worker '
                'environment or set INSIGHTFACE_CTX_ID=-1.'
            )
        return (
            ['CUDAExecutionProvider', 'CPUExecutionProvider'],
            [{'device_id': ctx_id}, {}],
        )
    return (['CPUExecutionProvider'], [{}])


def _provider_cache_name(base_name: str, ctx_id: int) -> str:
    stem, ext = os.path.splitext(base_name)
    provider = 'cuda' if ctx_id >= 0 else 'cpu'
    return f'{stem}_{provider}{ext}'


def _adaface_pack_dir() -> str:
    pack_dir = os.path.join(os.path.expanduser('~/.insightface'), 'models', MODEL_PACK)
    os.makedirs(pack_dir, exist_ok=True)
    return pack_dir


def _onnx_ready(path: str, min_bytes: int = 100_000) -> bool:
    return os.path.exists(path) and os.path.getsize(path) >= min_bytes


def _download_aux_onnx(pack_dir: str) -> None:
    """Download SCRFD + landmark ONNX into the adaface pack (no buffalo_l install)."""
    missing = [
        name for name in AUX_ONNX_FILES
        if not _onnx_ready(os.path.join(pack_dir, name))
    ]
    if not missing:
        return

    logger.info('Downloading AdaFace pipeline helpers (detector + landmarks)')
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, 'aux.zip')
        urlretrieve(AUX_ONNX_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            for name in missing:
                try:
                    zf.extract(name, pack_dir)
                except KeyError as exc:
                    raise FileNotFoundError(f'{name} not found in auxiliary model zip') from exc


def _download_adaface_recognition(pack_dir: str) -> None:
    # Always download the FP32 model — the INT8 variant is generated locally via
    # scripts/quantize_adaface.py and must never be fetched from Google Drive.
    fp32_path = os.path.join(pack_dir, FP32_ONNX)
    if _onnx_ready(fp32_path, min_bytes=1_000_000):
        return
    logger.info('Downloading AdaFace recognition model to %s', fp32_path)
    gdown.download(
        f'https://drive.google.com/uc?id={RECOG_GDRIVE_ID}',
        fp32_path,
        quiet=False,
    )


def _ensure_adaface_pack() -> str:
    """AdaFace-only pack: detector + landmarks + AdaFace IR101 recognition ONNX."""
    pack_dir = _adaface_pack_dir()
    _download_aux_onnx(pack_dir)
    _download_adaface_recognition(pack_dir)
    return pack_dir


def _patch_adaface_recognition(recog_model: ArcFaceONNX) -> None:
    """AdaFace expects BGR input; InsightFace ArcFaceONNX defaults to RGB (swapRB=True)."""

    def get_feat(imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        blob = cv2.dnn.blobFromImages(
            imgs,
            1.0 / recog_model.input_std,
            recog_model.input_size,
            (recog_model.input_mean,) * 3,
            swapRB=False,
        )
        return recog_model.session.run(
            recog_model.output_names, {recog_model.input_name: blob}
        )[0]

    recog_model.get_feat = get_feat


def _load_adaface_recognition(
    pack_dir: str,
    ctx_id: int,
    providers: list[str],
    provider_options: list[dict],
) -> ArcFaceONNX:
    recog_path = os.path.join(pack_dir, RECOG_ONNX)
    if not _onnx_ready(recog_path, min_bytes=100_000):
        if RECOG_ONNX != FP32_ONNX:
            raise FileNotFoundError(
                f'Configured model not found: {recog_path}\n'
                f'Generate it first:  python scripts/quantize_adaface.py'
            )
        raise FileNotFoundError(f'AdaFace model not found: {recog_path}')

    session = onnxruntime.InferenceSession(
        recog_path,
        providers=providers,
        provider_options=provider_options,
    )
    recog_model = ArcFaceONNX(recog_path, session=session)
    _patch_adaface_recognition(recog_model)
    recog_model.prepare(ctx_id)
    return recog_model


def _make_session_opts(save_cache_as: str = '') -> onnxruntime.SessionOptions:
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get('ORT_INTRA_THREADS', '0'))
    opts.inter_op_num_threads = int(os.environ.get('ORT_INTER_THREADS', '1'))
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    cache_dir = os.environ.get('ORT_OPTIMIZED_MODEL_DIR', '')
    if cache_dir and save_cache_as:
        os.makedirs(cache_dir, exist_ok=True)
        opts.optimized_model_filepath = os.path.join(cache_dir, save_cache_as)
    return opts


def _apply_session_opts(
    model_obj, sess_options: onnxruntime.SessionOptions, cache_name: str = ''
) -> None:
    """Rebuild model's InferenceSession with custom SessionOptions.

    On first run with ORT_OPTIMIZED_MODEL_DIR set, saves the optimized graph so
    subsequent startups load the pre-optimized model and skip the optimization phase.
    """
    old_session = model_obj.session
    providers = old_session.get_providers()
    provider_options = old_session.get_provider_options()
    po_list = [provider_options.get(p, {}) for p in providers]

    model_path = model_obj.model_file
    cache_dir = os.environ.get('ORT_OPTIMIZED_MODEL_DIR', '')
    using_cached_model = False
    if cache_dir and cache_name:
        cached = os.path.join(cache_dir, cache_name)
        if os.path.exists(cached) and os.path.getmtime(cached) >= os.path.getmtime(model_path):
            model_path = cached
            using_cached_model = True

    if using_cached_model:
        sess_options.optimized_model_filepath = ''

    model_obj.session = onnxruntime.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=providers,
        provider_options=po_list,
    )


def _warmup(face_app, recog_model) -> None:
    """Run dummy inference to pay ONNX kernel JIT cost at worker startup, not on first task."""
    try:
        face_app.det_model.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        recog_model.get_feat([np.zeros((112, 112, 3), dtype=np.uint8)])
        logger.info('ONNX warm-up completed')
    except Exception:
        pass


def _build_face_app():
    """AdaFace-only pipeline: SCRFD detect → landmarks → AdaFace IR101 embeddings."""
    pack_dir = _ensure_adaface_pack()
    ctx_id = int(os.environ.get('INSIGHTFACE_CTX_ID', '-1'))
    providers, provider_options = _execution_provider_config(ctx_id)

    face_app = insightface.app.FaceAnalysis(
        name=MODEL_PACK,
        allowed_modules=['detection', 'landmark_2d_106'],
        providers=providers,
        provider_options=provider_options,
    )
    det_thresh = float(os.environ.get('DET_THRESH', '0.5'))
    det_size_raw = os.environ.get('DET_SIZE', '640,640')
    det_size_parts = [int(part.strip()) for part in det_size_raw.split(',') if part.strip()]
    if len(det_size_parts) != 2:
        raise ValueError('DET_SIZE must be formatted as width,height, for example 640,640')
    face_app.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=tuple(det_size_parts))

    recog_model = _load_adaface_recognition(pack_dir, ctx_id, providers, provider_options)
    face_app.models['recognition'] = recog_model

    rec_cache = _provider_cache_name(os.path.splitext(RECOG_ONNX)[0] + '_opt.onnx', ctx_id)
    det_cache = _provider_cache_name('det_10g_opt.onnx', ctx_id)
    det_opts = _make_session_opts(det_cache)
    rec_opts = _make_session_opts(rec_cache)
    _apply_session_opts(face_app.det_model, det_opts, det_cache)
    _apply_session_opts(recog_model, rec_opts, rec_cache)

    _warmup(face_app, recog_model)
    logger.info(
        'AdaFace pipeline ready (ctx_id=%s, providers=%s, det_thresh=%s, det_size=%s)',
        ctx_id,
        face_app.det_model.session.get_providers(),
        det_thresh,
        tuple(det_size_parts),
    )
    return face_app


_app = None


def _get_app():
    global _app
    if _app is None:
        _app = _build_face_app()
    return _app


def extract_embeddings(image_path: str) -> list:
    """Return 512-d AdaFace embeddings (uses GPU-batched recognition path)."""
    results = extract_embeddings_batch([image_path])
    return results[0] if results else []


def extract_embeddings_batch(image_paths: list) -> list:
    """
    Batch AdaFace embeddings for multiple images.

    Detection runs per-image; recognition is batched across REC_BATCH_SIZE crops.
    Returns one inner list of embeddings per input path (same order).
    """
    face_app = _get_app()
    rec_model = face_app.models.get('recognition')
    if rec_model is None:
        return [extract_embeddings(p) for p in image_paths]

    image_size = rec_model.input_size[0]

    all_crops = []
    image_face_spans = []

    min_face_px = _min_face_px()

    for path in image_paths:
        img = cv2.imread(path)
        span_start = len(all_crops)

        if img is not None:
            bboxes, kpss = face_app.det_model.detect(img, max_num=0, metric='default')
            detected = int(bboxes.shape[0])
            kept = 0
            if bboxes.shape[0] > 0 and kpss is not None:
                for i in range(bboxes.shape[0]):
                    x1, y1, x2, y2 = bboxes[i, :4].astype(int)
                    if (x2 - x1) < min_face_px or (y2 - y1) < min_face_px:
                        continue
                    crop = _face_align.norm_crop(img, landmark=kpss[i], image_size=image_size)
                    all_crops.append(crop)
                    kept += 1
            logger.info(
                'Face detection path=%s decoded=%s detected=%d kept=%d min_face_px=%d',
                path,
                img.shape[:2],
                detected,
                kept,
                min_face_px,
            )
        else:
            logger.warning('OpenCV could not decode image path=%s', path)

        image_face_spans.append((span_start, len(all_crops)))

    if not all_crops:
        return [[] for _ in image_paths]

    batch_size = _rec_batch_size()
    all_embeddings = []
    for i in range(0, len(all_crops), batch_size):
        batch = all_crops[i:i + batch_size]
        embs = rec_model.get_feat(batch)
        all_embeddings.extend(embs)

    return [all_embeddings[start:end] for start, end in image_face_spans]


def _normalize_embedding_results(output, expected_count: int) -> list:
    if isinstance(output, list):
        results = output
    elif isinstance(output, dict):
        if 'embeddings' in output:
            results = output['embeddings']
        elif 'results' in output:
            raw_results = output['results']
            if not isinstance(raw_results, list):
                raise RuntimeError('RunPod output.results must be a list')
            results = [
                item.get('embeddings', item.get('faces', [])) if isinstance(item, dict) else item
                for item in raw_results
            ]
        else:
            results = [[] for _ in range(expected_count)]
    else:
        raise RuntimeError(f'Unexpected RunPod output type: {type(output).__name__}')

    if not isinstance(results, list):
        raise RuntimeError('RunPod embeddings output must be a list')
    if len(results) != expected_count:
        raise RuntimeError(
            f'RunPod returned {len(results)} result(s) for {expected_count} image(s)'
        )
    return results


def _call_runpod(images_payload: list) -> list:
    """Shared HTTP call to RunPod /runsync endpoint."""
    import requests as req_lib
    from django.conf import settings

    url = f'https://api.runpod.ai/v2/{settings.RUNPOD_ENDPOINT_ID}/runsync'
    headers = {
        'Authorization': f'Bearer {settings.RUNPOD_API_KEY}',
        'Content-Type': 'application/json',
    }
    payload = {'input': {'images': images_payload}}

    try:
        resp = req_lib.post(url, headers=headers, json=payload, timeout=settings.RUNPOD_TIMEOUT)
        resp.raise_for_status()
    except req_lib.exceptions.Timeout:
        logger.error('RunPod request timed out after %ss', settings.RUNPOD_TIMEOUT)
        raise
    except req_lib.exceptions.RequestException as exc:
        logger.error('RunPod HTTP error: %s', exc)
        raise

    data = resp.json()
    logger.info("here is the whole data",data)
    if data.get('status') != 'COMPLETED':
        error_msg = data.get('error', 'unknown RunPod error')
        logger.error('RunPod job not completed: status=%s error=%s', data.get('status'), error_msg)
        raise RuntimeError(f'RunPod inference failed: {error_msg}')

    output = data.get('output', {})
    results = _normalize_embedding_results(output, len(images_payload))
    logger.info('here is the result', results)
    logger.info("here is the output",output)
    logger.info('RunPod face counts: %s', [len(item) for item in results])
    return results


def extract_embeddings_runpod_urls(image_urls: list) -> list:
    """
    Dispatch inference to RunPod GPU using public image URLs (e.g. Cloudflare R2).

    The RunPod container fetches images directly from the URLs — no download
    to the Celery worker. Use this when USE_R2_STORAGE=True.

    Returns same format as extract_embeddings_batch().
    """
    logger.info('RunPod inference (URLs) for %d images', len(image_urls))
    return _call_runpod([{'url': u} for u in image_urls])


def extract_embeddings_runpod(image_paths: list) -> list:
    """
    Dispatch inference to RunPod GPU by base64-encoding local image files.

    Fallback for local dev when USE_R2_STORAGE=False and there are no public URLs.

    Returns same format as extract_embeddings_batch().
    """
    import base64

    images_payload = []
    for path in image_paths:
        with open(path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('ascii')
        images_payload.append({'b64': b64})

    logger.info('RunPod inference (base64) for %d images', len(image_paths))
    return _call_runpod(images_payload)


def find_similar(query_embedding, event_id=None, threshold=0.45, limit=50):
    """
    Use pgvector cosine distance to find photos containing a matching face.

    query_embedding: np.ndarray (512,)
    event_id: optional int — scope search to a single event
    threshold: cosine similarity floor (0–1); higher = stricter match
    Returns list of Photo objects ordered by closest match.
    """
    from pgvector.django import CosineDistance
    from django.db.models import Min
    from .models import FaceEmbedding, Photo

    max_distance = 1.0 - threshold
    vec = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding

    qs = FaceEmbedding.objects.annotate(
        distance=CosineDistance('embedding', vec)
    ).filter(distance__lt=max_distance)

    if event_id:
        qs = qs.filter(photo__album__event_id=event_id)

    best_per_photo = (
        qs.values('photo_id')
          .annotate(min_distance=Min('distance'))
          .order_by('min_distance')
    )

    if not best_per_photo:
        return []

    dist_map = {r['photo_id']: r['min_distance'] for r in best_per_photo[:limit]}
    photos = Photo.objects.filter(id__in=dist_map.keys()).select_related('album')
    return sorted(photos, key=lambda p: dist_map[p.id])[:limit]
