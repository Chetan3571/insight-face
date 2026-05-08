from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .models import Photo, Album
from .face_utils import extract_embeddings, find_similar
import tempfile, os


def home(request):
    """Small frontend for upload and face search."""
    return render(request, 'event/index.html')


def list_albums(request):
    """GET: return available albums for the frontend selector."""
    if request.method != 'GET':
        return JsonResponse({'error': 'GET only'}, status=405)

    albums = Album.objects.select_related('event').all().order_by('-id')
    return JsonResponse({
        'albums': [
            {
                'id': album.id,
                'title': album.title,
                'event': album.event.name,
            }
            for album in albums
        ]
    })


@csrf_exempt
def upload_photo(request):
    """POST: upload multiple photos to an album → auto-extract face embeddings."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    album_id = request.POST.get('album_id')
    images = request.FILES.getlist('images')

    print('id',album_id)
    if not album_id or not images:
        return JsonResponse({'error': 'album_id and at least one image are required'}, status=400)

    try:
        album = Album.objects.get(id=album_id)
        print(album)
    except Album.DoesNotExist:
        return JsonResponse({'error': f'Album {album_id} not found'}, status=404)

    count =1
    results = []
    for image in images:
        photo = Photo(album=album, image=image)
        photo.save()
        print(count)
        count+=1
        embeddings = extract_embeddings(photo.image.path)
        photo.set_embeddings(embeddings)
        photo.save()

        results.append({
            'id': photo.id,
            'filename': image.name,
            'faces_found': len(embeddings),
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

    # Save to a temp file so InsightFace can read it
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        for chunk in query_image.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        embeddings = extract_embeddings(tmp_path)
        if not embeddings:
            return JsonResponse({'error': 'No face detected in query image'}, status=400)

        query_emb = embeddings[0]   # use the first detected face
        all_photos = Photo.objects.exclude(face_embeddings='[]')
        matched = find_similar(query_emb, all_photos)
        print(matched)
        # count_photos =matched.count()
        #     # 'photos':count_photos,
        # print(count_photos)
        return JsonResponse({
            'matched_count': len(matched),

            'matched_photos': [
                {'id': p.id, 'url': p.image.url, 'album': p.album.title}
                for p in matched
            ]
        })
    finally:
        os.unlink(tmp_path)
