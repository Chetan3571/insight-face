from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('albums/', views.list_albums, name='list_albums'),
    path('upload/', views.upload_photo),
    path('search/', views.search_by_face),
]