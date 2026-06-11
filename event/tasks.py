import logging

from celery import shared_task
from django.conf import settings

from .cache_utils import invalidate_search_cache
from .models import FaceEmbedding, Photo

logger = logging.getLogger('event')


def _save_photo_embeddings(photo, embeddings) -> int:
    FaceEmbedding.objects.filter(photo=photo).delete()
    if embeddings:
        FaceEmbedding.objects.bulk_create([
            FaceEmbedding(
                photo=photo,
                embedding=emb.tolist() if hasattr(emb, 'tolist') else emb,
            )
            for emb in embeddings
        ])
    photo.processed = True
    photo.save(update_fields=['processed'])
    return len(embeddings)


@shared_task(bind=True, max_retries=3, default_retry_delay=5)
def process_photo_embeddings(self, photo_id):
    """Extract face embeddings for a single photo and store them via pgvector."""
    from .face_utils import extract_embeddings_batch

    try:
        photo = Photo.objects.get(id=photo_id)
    except Photo.DoesNotExist:
        logger.error('Photo %s not found for embedding task', photo_id)
        return {'photo_id': photo_id, 'error': 'photo not found'}

    try:
        batch = extract_embeddings_batch([photo.image.path])
        embeddings = batch[0] if batch else []
        faces_found = _save_photo_embeddings(photo, embeddings)
        invalidate_search_cache()

        logger.info('Photo %s processed: %s faces', photo_id, faces_found)
        return {'photo_id': photo_id, 'faces_found': faces_found}
    except FileNotFoundError as exc:
        logger.error('Image missing for photo %s: %s', photo_id, exc)
        return {'photo_id': photo_id, 'error': 'image file not found'}
    except Exception as exc:
        logger.exception('Embedding extraction failed for photo %s', photo_id)
        raise self.retry(exc=exc)


@shared_task
def process_album_embeddings(album_id):
    """
    GPU-batched embedding extraction for an entire album.

    Photos are grouped into ALBUM_PHOTO_BATCH_SIZE chunks; all face crops
    within each chunk are forwarded to the recognition model in a single
    ONNX forward pass (REC_BATCH_SIZE crops at a time).  Falls back to
    per-photo Celery tasks for any chunk that raises.
    """
    from .face_utils import extract_embeddings_batch
    from .models import Album

    photos = list(Album.objects.get(id=album_id).photos.all())
    batch_size = getattr(settings, 'ALBUM_PHOTO_BATCH_SIZE', 8)
    processed = 0

    for i in range(0, len(photos), batch_size):
        chunk = photos[i:i + batch_size]
        paths = [p.image.path for p in chunk]

        try:
            batch_results = extract_embeddings_batch(paths)
        except Exception:
            logger.exception(
                'Batch failed for album %s chunk at index %s — falling back to per-photo tasks',
                album_id, i,
            )
            for p in chunk:
                process_photo_embeddings.delay(p.id)
            continue

        for photo, embeddings in zip(chunk, batch_results):
            _save_photo_embeddings(photo, embeddings)
            processed += 1

        logger.info('Album %s: batch %s–%s done', album_id, i, i + len(chunk) - 1)

    invalidate_search_cache()
    return {'album_id': album_id, 'photos_processed': processed}
