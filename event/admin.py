from django.contrib import admin
from django.db.models import Count

from .models import Album, Event, FaceEmbedding, Photo


def embedding_preview(vec, n=8):
    if vec is None:
        return ''
    values = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
    head = ', '.join(f'{v:.4f}' for v in values[:n])
    if len(values) > n:
        return f'[{head}, ...] ({len(values)} dims)'
    return f'[{head}]'


class FaceEmbeddingInline(admin.TabularInline):
    model = FaceEmbedding
    extra = 0
    can_delete = False
    readonly_fields = ('embedding_preview',)
    fields = ('id', 'embedding_preview')

    def has_add_permission(self, request, obj=None):
        return False

    @admin.display(description='Embedding')
    def embedding_preview(self, obj):
        return embedding_preview(obj.embedding)


@admin.register(Event)
class EventAdmin(admin.ModelAdmin):
    list_display = ('name', 'date', 'location')
    list_filter = ('date',)
    search_fields = ('name', 'location')


@admin.register(Album)
class AlbumAdmin(admin.ModelAdmin):
    list_display = ('title', 'event', 'photo_count')
    list_filter = ('event',)
    search_fields = ('title', 'event__name')

    def get_queryset(self, request):
        return super().get_queryset(request).annotate(photo_count=Count('photos'))

    @admin.display(description='Photos', ordering='photo_count')
    def photo_count(self, obj):
        return obj.photo_count


@admin.register(Photo)
class PhotoAdmin(admin.ModelAdmin):
    list_display = ('id', 'album', 'processed', 'face_count', 'image')
    list_filter = ('processed', 'album__event', 'album')
    search_fields = ('id', 'album__title', 'image')
    inlines = (FaceEmbeddingInline,)

    def get_queryset(self, request):
        return super().get_queryset(request).annotate(face_count=Count('face_embeddings'))

    @admin.display(description='Faces', ordering='face_count')
    def face_count(self, obj):
        return obj.face_count


@admin.register(FaceEmbedding)
class FaceEmbeddingAdmin(admin.ModelAdmin):
    list_display = ('id', 'photo', 'embedding_preview')
    list_filter = ('photo__album__event', 'photo__album')
    search_fields = ('photo__id',)
    readonly_fields = ('photo', 'embedding_preview', 'embedding_full')

    def has_add_permission(self, request):
        return False

    @admin.display(description='Preview')
    def embedding_preview(self, obj):
        return embedding_preview(obj.embedding)

    @admin.display(description='Full embedding')
    def embedding_full(self, obj):
        if obj.embedding is None:
            return ''
        values = obj.embedding.tolist() if hasattr(obj.embedding, 'tolist') else list(obj.embedding)
        return ', '.join(f'{v:.6f}' for v in values)
