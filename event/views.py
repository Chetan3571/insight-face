from django.db import connection
from django.http import JsonResponse
from django.shortcuts import render

from .models import Album, Event


def home(request):
    return render(request, 'event/index.html')


def health(request):
    payload = {'status': 'ok', 'database': 'ok'}
    try:
        with connection.cursor() as cursor:
            cursor.execute('SELECT 1')
    except Exception as exc:
        payload['status'] = 'error'
        payload['database'] = str(exc)
        return JsonResponse(payload, status=503)
    return JsonResponse(payload)


def list_events(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'GET only'}, status=405)

    events = Event.objects.all().order_by('-date', '-id')
    return JsonResponse({
        'events': [
            {'id': e.id, 'name': e.name, 'date': e.date.isoformat(), 'location': e.location}
            for e in events
        ],
    })


def list_albums(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'GET only'}, status=405)

    albums = Album.objects.select_related('event').all().order_by('-id')
    return JsonResponse({
        'albums': [
            {'id': a.id, 'title': a.title, 'event': a.event.name, 'event_id': a.event_id}
            for a in albums
        ],
    })
