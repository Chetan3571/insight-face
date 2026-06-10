from django.urls import path

from . import views
from .api_views import SearchByFaceAPIView, UploadPhotoAPIView

urlpatterns = [
    path('', views.home, name='home'),
    path('health/', views.health, name='health'),
    path('albums/', views.list_albums, name='list_albums'),
    path('upload/', UploadPhotoAPIView.as_view(), name='upload_photo'),
    path('search/', SearchByFaceAPIView.as_view(), name='search_by_face'),
]