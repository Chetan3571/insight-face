from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_photo),
    path('search/', views.search_by_face),
]