import insightface
import numpy as np
import cv2

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
