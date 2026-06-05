from django.http import JsonResponse
from django.shortcuts import render

from .models import Album


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
