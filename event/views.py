from django.http import JsonResponse
from django.shortcuts import render

from .models import Album


def home(request):
    return render(request, 'event/index.html')


def list_albums(request):
    albums = Album.objects.select_related('event').all()
    return JsonResponse({
        'albums': [
            {'id': a.id, 'title': a.title, 'event': a.event.name}
            for a in albums
        ],
    })
