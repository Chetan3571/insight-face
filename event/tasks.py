import logging

from celery import shared_task

from .models import FaceEmbedding, Photo

logger = logging.getLogger('event')


@shared_task(bind=True, max_retries=3, default_retry_delay=5)
def process_photo_embeddings(self, photo_id):
    """Extract face embeddings for a single photo and store them via pgvector."""
    from .face_utils import extract_embeddings

    try:
        photo = Photo.objects.get(id=photo_id)
    except Photo.DoesNotExist:
        logger.error('Photo %s not found for embedding task', photo_id)
        return {'photo_id': photo_id, 'error': 'photo not found'}

    try:
        image_path = photo.image.path
        embeddings = extract_embeddings(image_path)

        FaceEmbedding.objects.filter(photo=photo).delete()
        FaceEmbedding.objects.bulk_create([
            FaceEmbedding(photo=photo, embedding=emb.tolist())
            for emb in embeddings
        ])

        photo.processed = True
        photo.save(update_fields=['processed'])

        logger.info('Photo %s processed: %s faces', photo_id, len(embeddings))
        return {'photo_id': photo_id, 'faces_found': len(embeddings)}
    except FileNotFoundError as exc:
        logger.error('Image missing for photo %s: %s', photo_id, exc)
        return {'photo_id': photo_id, 'error': 'image file not found'}
    except Exception as exc:
        logger.exception('Embedding extraction failed for photo %s', photo_id)
        raise self.retry(exc=exc)


@shared_task
def process_album_embeddings(album_id):
    """Enqueue embedding tasks for every photo in an album."""
    from .models import Album

    photo_ids = list(
        Album.objects.get(id=album_id).photos.values_list('id', flat=True)
    )
    for photo_id in photo_ids:
        process_photo_embeddings.delay(photo_id)
    return {'album_id': album_id, 'photos_queued': len(photo_ids)}
