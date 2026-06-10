import os
import shutil

import cv2
import gdown
import insightface
import numpy as np
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from insightface.utils.storage import ensure_available

MODEL_PACK = 'adaface'
RECOG_ONNX = 'adaface_ir101_webface12m.onnx'
RECOG_GDRIVE_ID = '1dgMFOASKnaujQcCL4sSYkKOkBrmXUUU1'
# Detection/alignment from buffalo_l; recognition is AdaFace IR101 WebFace12M
BUFFALO_SYMLINKS = ('det_10g.onnx', '2d106det.onnx')


def _ensure_adaface_pack():
    """Build ~/.insightface/models/adaface from buffalo_l + AdaFace recognition ONNX."""
    root = os.path.expanduser('~/.insightface')
    pack_dir = os.path.join(root, 'models', MODEL_PACK)
    os.makedirs(pack_dir, exist_ok=True)

    buffalo_dir = ensure_available('models', 'buffalo_l', root=root)
    for fname in BUFFALO_SYMLINKS:
        src = os.path.join(buffalo_dir, fname)
        dst = os.path.join(pack_dir, fname)
        if not os.path.exists(dst):
            if os.path.exists(src):
                os.symlink(src, dst)
            else:
                shutil.copy2(src, dst)

    recog_path = os.path.join(pack_dir, RECOG_ONNX)
    if not os.path.exists(recog_path) or os.path.getsize(recog_path) < 1_000_000:
        gdown.download(
            f'https://drive.google.com/uc?id={RECOG_GDRIVE_ID}',
            recog_path,
            quiet=False,
        )

    return pack_dir


def _patch_adaface_recognition(recog_model):
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


app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)

MIN_FACE_PX = 40  # ignore faces smaller than 40×40 px (too blurry to be reliable)


def extract_embeddings(image_path: str) -> list:
    """Return 512-d embeddings for every adequately-sized face in the image."""
    img = cv2.imread(image_path)
    if img is None:
        return []
    faces = app.get(img)
    result = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        if (x2 - x1) < MIN_FACE_PX or (y2 - y1) < MIN_FACE_PX:
            continue
        result.append(face.embedding)
    return result


def find_similar(query_embedding, event_id=None, threshold=0.45):
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

    # Best match distance per photo
    best_per_photo = (
        qs.values('photo_id')
          .annotate(min_distance=Min('distance'))
          .order_by('min_distance')
    )

    if not best_per_photo:
        return []

    dist_map = {r['photo_id']: r['min_distance'] for r in best_per_photo}
    photos = Photo.objects.filter(id__in=dist_map.keys()).select_related('album')
    return sorted(photos, key=lambda p: dist_map[p.id])
