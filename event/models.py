from django.db import models
import json

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
    # Stores list of face embeddings as JSON (one per detected face)
    face_embeddings = models.TextField(blank=True, default='[]')

    def get_embeddings(self):
        return json.loads(self.face_embeddings)

    def set_embeddings(self, embeddings):
        # embeddings: list of lists (each a 512-d float vector)
        self.face_embeddings = json.dumps(
            [e.tolist() if hasattr(e, 'tolist') else e for e in embeddings]
        )