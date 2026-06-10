from django.urls import path

from . import views
from .api_views import SearchByFaceAPIView, UploadPhotoAPIView, home, list_albums

urlpatterns = [
    path('', home, name='home'),
    path('albums/', list_albums, name='list_albums'),
    path('upload/', UploadPhotoAPIView.as_view(), name='upload_photo'),
    path('search/', SearchByFaceAPIView.as_view(), name='search_by_face'),
]