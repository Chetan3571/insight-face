import insightface
import numpy as np
import cv2

# Load model once at module level (buffalo_l is the recommended default)
app = insightface.app.FaceAnalysis(name='buffalo_l')
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