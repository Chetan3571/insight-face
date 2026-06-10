import logging
import os
import tempfile
import zipfile
from urllib.request import urlretrieve

import cv2
import gdown
import insightface
import numpy as np
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from insightface.utils import face_align as _face_align

logger = logging.getLogger('event')

MODEL_PACK = 'adaface'
RECOG_ONNX = 'adaface_ir101_webface12m.onnx'
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
REC_BATCH_SIZE = 16


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
    recog_path = os.path.join(pack_dir, RECOG_ONNX)
    if _onnx_ready(recog_path, min_bytes=1_000_000):
        return
    logger.info('Downloading AdaFace recognition model to %s', recog_path)
    gdown.download(
        f'https://drive.google.com/uc?id={RECOG_GDRIVE_ID}',
        recog_path,
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


def _load_adaface_recognition(pack_dir: str, ctx_id: int) -> ArcFaceONNX:
    recog_path = os.path.join(pack_dir, RECOG_ONNX)
    if not _onnx_ready(recog_path, min_bytes=1_000_000):
        raise FileNotFoundError(f'AdaFace model not found: {recog_path}')

    recog_model = ArcFaceONNX(recog_path)
    _patch_adaface_recognition(recog_model)
    recog_model.prepare(ctx_id)
    return recog_model


def _build_face_app():
    """AdaFace-only pipeline: SCRFD detect → landmarks → AdaFace IR101 embeddings."""
    pack_dir = _ensure_adaface_pack()
    ctx_id = int(os.environ.get('INSIGHTFACE_CTX_ID', '-1'))

    face_app = insightface.app.FaceAnalysis(
        name=MODEL_PACK,
        allowed_modules=['detection', 'landmark_2d_106'],
    )
    face_app.prepare(ctx_id=ctx_id)

    recog_model = _load_adaface_recognition(pack_dir, ctx_id)
    face_app.models['recognition'] = recog_model
    logger.info('AdaFace pipeline ready (ctx_id=%s)', ctx_id)
    return face_app


app = _build_face_app()


def extract_embeddings(image_path: str) -> list:
    """Return 512-d AdaFace embeddings for every adequately-sized face in the image."""
    img = cv2.imread(image_path)
    if img is None:
        return []
    faces = app.get(img)
    result = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        if (x2 - x1) < MIN_FACE_PX or (y2 - y1) < MIN_FACE_PX:
            continue
        if face.embedding is None:
            continue
        result.append(face.embedding)
    return result


def extract_embeddings_batch(image_paths: list) -> list:
    """
    Batch AdaFace embeddings for multiple images.

    Detection runs per-image; recognition is batched across REC_BATCH_SIZE crops.
    Returns one inner list of embeddings per input path (same order).
    """
    rec_model = app.models.get('recognition')
    if rec_model is None:
        return [extract_embeddings(p) for p in image_paths]

    image_size = rec_model.input_size[0]

    all_crops = []
    image_face_spans = []

    for path in image_paths:
        img = cv2.imread(path)
        span_start = len(all_crops)

        if img is not None:
            bboxes, kpss = app.det_model.detect(img, max_num=0, metric='default')
            if bboxes.shape[0] > 0 and kpss is not None:
                for i in range(bboxes.shape[0]):
                    x1, y1, x2, y2 = bboxes[i, :4].astype(int)
                    if (x2 - x1) < MIN_FACE_PX or (y2 - y1) < MIN_FACE_PX:
                        continue
                    crop = _face_align.norm_crop(img, landmark=kpss[i], image_size=image_size)
                    all_crops.append(crop)

        image_face_spans.append((span_start, len(all_crops)))

    if not all_crops:
        return [[] for _ in image_paths]

    all_embeddings = []
    for i in range(0, len(all_crops), REC_BATCH_SIZE):
        batch = all_crops[i:i + REC_BATCH_SIZE]
        embs = rec_model.get_feat(batch)
        all_embeddings.extend(embs)

    return [all_embeddings[start:end] for start, end in image_face_spans]


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
