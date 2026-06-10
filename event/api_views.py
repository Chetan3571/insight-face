import logging
import os
import tempfile
import time
import traceback

from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .face_utils import extract_embeddings, find_similar
from .models import Album, Photo
from .serializers import (
    SearchByFaceResponseSerializer,
    SearchByFaceSerializer,
    UploadPhotoResponseSerializer,
    UploadPhotoSerializer,
)

from django.http import JsonResponse
from django.shortcuts import render

from .models import Album
logger = logging.getLogger('event')


class UploadPhotoAPIView(APIView):

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        start = time.time()
        logger.info('Upload started')

        try:
            serializer = UploadPhotoSerializer(data=request.data)
            serializer.is_valid(raise_exception=True)

            album_id = serializer.validated_data['album_id']
            image = serializer.validated_data['image']
            logger.info('Upload validated album_id=%s filename=%s size=%s', album_id, image.name, image.size)

            try:
                album = Album.objects.get(id=album_id)
            except Album.DoesNotExist:
                logger.warning('Album %s not found', album_id)
                return Response(
                    {'error': f'Album {album_id} not found'},
                    status=status.HTTP_404_NOT_FOUND,
                )

            photo = Photo(album=album, image=image)
            photo.save()
            logger.info('Photo saved id=%s path=%s', photo.id, photo.image.path)

            embeddings = extract_embeddings(photo.image.path)
            logger.info('Embeddings extracted count=%s', len(embeddings))

            photo.set_embeddings(embeddings)
            photo.save()

            elapsed = time.time() - start
            logger.info('Upload completed id=%s in %.2fs', photo.id, elapsed)

            response_data = {
                'id': photo.id,
                'filename': image.name,
                'faces_found': len(embeddings),
            }
            return Response(
                UploadPhotoResponseSerializer(response_data).data,
                status=status.HTTP_201_CREATED,
            )
        except Exception:
            logger.error('Upload failed after %.2fs', time.time() - start)
            return Response(
                {'error': 'Upload failed', 'detail': traceback.format_exc()},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class SearchByFaceAPIView(APIView):
    """POST /api/search/ — find photos containing a face matching the query image."""

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        serializer = SearchByFaceSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        query_image = serializer.validated_data['image']

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            for chunk in query_image.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            embeddings = extract_embeddings(tmp_path)
            if not embeddings:
                return Response(
                    {'error': 'No face detected in query image'},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            query_emb = embeddings[0]
            all_photos = Photo.objects.exclude(face_embeddings='[]')
            matched = find_similar(query_emb, all_photos)

            response_data = {
                'matched_count': len(matched),
                'matched_photos': [
                    {'id': p.id, 'url': p.image.url, 'album': p.album.title}
                    for p in matched
                ],
            }
            return Response(SearchByFaceResponseSerializer(response_data).data)
        finally:
            os.unlink(tmp_path)


def home(request):
    return render(request, 'event/index.html')


def list_albums(request):
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
