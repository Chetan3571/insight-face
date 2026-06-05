import os
import tempfile

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


class UploadPhotoAPIView(APIView):
    """POST /api/upload/ — upload photos to an album and extract face embeddings."""

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        serializer = UploadPhotoSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        images = request.FILES.getlist('images')
        if not images:
            return Response(
                {'error': 'album_id and at least one image are required'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        album_id = serializer.validated_data['album_id']
        try:
            album = Album.objects.get(id=album_id)
        except Album.DoesNotExist:
            return Response(
                {'error': f'Album {album_id} not found'},
                status=status.HTTP_404_NOT_FOUND,
            )

        results = []
        for image in images:
            photo = Photo(album=album, image=image)
            photo.save()
            embeddings = extract_embeddings(photo.image.path)
            photo.set_embeddings(embeddings)
            photo.save()

            results.append({
                'id': photo.id,
                'filename': image.name,
                'faces_found': len(embeddings),
            })

        response_data = {'uploaded': len(results), 'photos': results}
        return Response(
            UploadPhotoResponseSerializer(response_data).data,
            status=status.HTTP_201_CREATED,
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
