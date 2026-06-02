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


_ensure_adaface_pack()
app = insightface.app.FaceAnalysis(name=MODEL_PACK)
if isinstance(app.models.get('recognition'), ArcFaceONNX):
    _patch_adaface_recognition(app.models['recognition'])
app.prepare(ctx_id=0)   # ctx_id=0 → GPU if available, else CPU


def extract_embeddings(image_path: str) -> list:
    """Returns a list of 512-d embeddings, one per face found."""
    img = cv2.imread(image_path)
    if img is None:
        return []
    faces = app.get(img)
    return [face.embedding for face in faces]   # each is np.ndarray shape (512,)


def find_similar(query_embedding, all_photos, threshold=0.45):
    """
    query_embedding: np.ndarray (512,) from a query image
    all_photos: queryset of Photo objects
    Returns Photo objects where cosine similarity > threshold
    """
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    matches = []

    for photo in all_photos:
        embeddings = photo.get_embeddings()
        for emb in embeddings:
            emb = np.array(emb)
            emb_norm = emb / np.linalg.norm(emb)
            similarity = float(np.dot(query_norm, emb_norm))
            if similarity > threshold:
                matches.append((photo, similarity))
                break   # one face match per photo is enough

    matches.sort(key=lambda x: x[1], reverse=True)
    return [photo for photo, _ in matches]
