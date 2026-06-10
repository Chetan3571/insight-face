from django.db import models
from pgvector.django import VectorField, HnswIndex


class Event(models.Model):
    name = models.CharField(max_length=255)
    date = models.DateField()
    location = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return self.name


class Album(models.Model):
    event = models.ForeignKey(Event, on_delete=models.CASCADE, related_name='albums')
    title = models.CharField(max_length=255)

    def __str__(self):
        return self.title


class Photo(models.Model):
    album = models.ForeignKey(Album, on_delete=models.CASCADE, related_name='photos')
    image = models.ImageField(upload_to='photos/')
    processed = models.BooleanField(default=False)

    def __str__(self):
        return f'Photo {self.id} ({self.album})'


class FaceEmbedding(models.Model):
    photo = models.ForeignKey(Photo, on_delete=models.CASCADE, related_name='face_embeddings')
    embedding = VectorField(dimensions=512)

    class Meta:
        indexes = [
            HnswIndex(
                fields=['embedding'],
                name='face_emb_hnsw_idx',
                m=16,
                ef_construction=64,
                opclasses=['vector_cosine_ops'],
            )
        ]
