from django.urls import path

from . import views
from .api_views import SearchByFaceAPIView, TaskStatusAPIView, UploadPhotoAPIView

urlpatterns = [
    path('', views.home, name='home'),
    path('health/', views.health, name='health'),
    path('events/', views.list_events, name='list_events'),
    path('albums/', views.list_albums, name='list_albums'),
    path('albums/<int:album_id>/process/', views.process_album, name='process_album'),
    path('upload/', UploadPhotoAPIView.as_view(), name='upload_photo'),
    path('search/', SearchByFaceAPIView.as_view(), name='search_by_face'),
    path('tasks/<str:task_id>/', TaskStatusAPIView.as_view(), name='task_status'),
]