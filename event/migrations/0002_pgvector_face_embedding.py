import django.db.models.deletion
from django.db import migrations, models
from pgvector.django import VectorField, HnswIndex


class Migration(migrations.Migration):

    dependencies = [
        ('event', '0001_initial'),
    ]

    operations = [
        # Enable the pgvector extension (idempotent)
        migrations.RunSQL(
            sql='CREATE EXTENSION IF NOT EXISTS vector',
            reverse_sql='DROP EXTENSION IF EXISTS vector',
        ),

        # Drop the old JSON-blob column
        migrations.RemoveField(
            model_name='photo',
            name='face_embeddings',
        ),

        # Track whether embedding extraction has finished
        migrations.AddField(
            model_name='photo',
            name='processed',
            field=models.BooleanField(default=False),
        ),

        # One row per detected face — enables pgvector ANN search
        migrations.CreateModel(
            name='FaceEmbedding',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('photo', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='face_embeddings',
                    to='event.photo',
                )),
                ('embedding', VectorField(dimensions=512)),
            ],
        ),

        # HNSW index for fast cosine-distance ANN queries
        migrations.AddIndex(
            model_name='faceembedding',
            index=HnswIndex(
                fields=['embedding'],
                name='face_emb_hnsw_idx',
                m=16,
                ef_construction=64,
                opclasses=['vector_cosine_ops'],
            ),
        ),
    ]
