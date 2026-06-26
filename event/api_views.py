import hashlib
import logging
import os
import tempfile
import time

from django.conf import settings
from django.core.cache import cache
from rest_framework import status
from rest_framework.exceptions import ValidationError
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Album, Photo
from .serializers import (
    SearchByFaceResponseSerializer,
    SearchByFaceSerializer,
    UploadPhotoSerializer,
)
from .tasks import process_photo_embeddings

logger = logging.getLogger('event')


def _extract_search_embeddings(image_path):
    if settings.USE_RUNPOD:
        from .face_utils import extract_embeddings_runpod

        batch = extract_embeddings_runpod([image_path])
        if not isinstance(batch, list):
            raise RuntimeError(
                f'RunPod search extraction returned {type(batch).__name__}, expected list'
            )
        if len(batch) != 1:
            raise RuntimeError(
                f'RunPod search extraction returned {len(batch)} result(s) for 1 image'
            )
        return batch[0] if batch else []

    from .face_utils import extract_embeddings

    return extract_embeddings(image_path)


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
            logger.info(
                'Upload validated album_id=%s filename=%s size=%s',
                album_id, image.name, image.size,
            )

            try:
                album = Album.objects.get(id=album_id)
            except Album.DoesNotExist:
                logger.warning('Album %s not found', album_id)
                return Response(
                    {'error': f'Album {album_id} not found'},
                    status=status.HTTP_404_NOT_FOUND,
                )

            photo = Photo.objects.create(album=album, image=image)
            task = process_photo_embeddings.delay(photo.id)
            logger.info('Photo saved id=%s, celery task=%s', photo.id, task.id)

            return Response(
                {
                    'id': photo.id,
                    'filename': image.name,
                    'task_id': task.id,
                    'status': 'processing',
                },
                status=status.HTTP_201_CREATED,
            )
        except ValidationError as exc:
            return Response(exc.detail, status=status.HTTP_400_BAD_REQUEST)
        except Exception:
            logger.exception('Upload failed after %.2fs', time.time() - start)
            return Response(
                {'error': 'Upload failed'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class SearchByFaceAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        from .face_utils import find_similar

        serializer = SearchByFaceSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        query_image = serializer.validated_data['image']
        event_id = serializer.validated_data.get('event_id')

        query_image.seek(0)
        image_bytes = query_image.read()
        digest = hashlib.sha256(image_bytes).hexdigest()
        cache_key = f'search:{digest}:{event_id or "all"}'

        cached = cache.get(cache_key)
        if cached is not None:
            logger.info('Search cache hit key=%s', cache_key)
            return Response(SearchByFaceResponseSerializer(cached).data)

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        try:
            embeddings = _extract_search_embeddings(tmp_path)
            if not embeddings:
                return Response(
                    {'error': 'No face detected in query image'},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            matched = find_similar(embeddings[0], event_id=event_id)
            response_data = {
                'matched_count': len(matched),
                'matched_photos': [
                    {'id': p.id, 'url': p.image.url, 'album': p.album.title}
                    for p in matched
                ],
            }

            ttl = getattr(settings, 'SEARCH_CACHE_TTL', 300)
            cache.set(cache_key, response_data, ttl)

            return Response(SearchByFaceResponseSerializer(response_data).data)
        except Exception:
            logger.exception('Face search failed')
            return Response(
                {'error': 'Face search failed'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        finally:
            os.unlink(tmp_path)


class TaskStatusAPIView(APIView):
    def get(self, request, task_id):
        from django_celery_results.models import TaskResult

        try:
            result = TaskResult.objects.get(task_id=task_id)
            return Response({
                'task_id': task_id,
                'status': result.status,
                'result': result.result,
            })
        except TaskResult.DoesNotExist:
            return Response({
                'task_id': task_id,
                'status': 'PENDING',
            })
