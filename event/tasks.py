from celery import shared_task
from .models import Photo, FaceEmbedding


@shared_task(bind=True, max_retries=3, default_retry_delay=5)
def process_photo_embeddings(self, photo_id):
    """Extract face embeddings for a single photo and store them via pgvector."""
    from .face_utils import extract_embeddings
    try:
        photo = Photo.objects.get(id=photo_id)
        embeddings = extract_embeddings(photo.image.path)

        FaceEmbedding.objects.filter(photo=photo).delete()
        FaceEmbedding.objects.bulk_create([
            FaceEmbedding(photo=photo, embedding=emb.tolist())
            for emb in embeddings
        ])

        photo.processed = True
        photo.save(update_fields=['processed'])

        return {'photo_id': photo_id, 'faces_found': len(embeddings)}
    except Exception as exc:
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
