from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Photo, Album
from .face_utils import extract_embeddings, find_similar
from .tasks import process_photo_embeddings
import tempfile
import os

from .models import Album

@csrf_exempt
def upload_photo(request):
    """POST: upload photos to an album; embedding extraction runs in background via Celery."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    album_id = request.POST.get('album_id')
    images = request.FILES.getlist('images')

    if not album_id or not images:
        return JsonResponse({'error': 'album_id and at least one image are required'}, status=400)

    try:
        album = Album.objects.get(id=album_id)
    except Album.DoesNotExist:
        return JsonResponse({'error': f'Album {album_id} not found'}, status=404)

    results = []
    for image in images:
        photo = Photo.objects.create(album=album, image=image)
        task = process_photo_embeddings.delay(photo.id)
        results.append({
            'id': photo.id,
            'filename': image.name,
            'task_id': task.id,
            'status': 'processing',
        })

    return JsonResponse({'uploaded': len(results), 'photos': results})


@csrf_exempt
def search_by_face(request):
    """POST: upload a face image → returns all photos containing that face."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    query_image = request.FILES.get('image')
    if not query_image:
        return JsonResponse({'error': 'image is required'}, status=400)

    event_id = request.POST.get('event_id')

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        for chunk in query_image.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        embeddings = extract_embeddings(tmp_path)
        if not embeddings:
            return JsonResponse({'error': 'No face detected in query image'}, status=400)

        matched = find_similar(embeddings[0], event_id=event_id)
        return JsonResponse({
            'matched_count': len(matched),
            'matched_photos': [
                {'id': p.id, 'url': p.image.url, 'album': p.album.title}
                for p in matched
            ],
        })
    finally:
        os.unlink(tmp_path)
