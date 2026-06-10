import django.db.models.deletion
from django.db import migrations, models
from pgvector.django import VectorField, HnswIndex


def _ensure_vector_extension(apps, schema_editor):
    """Create pgvector if missing. On AWS RDS, master user must enable it first."""
    with schema_editor.connection.cursor() as cursor:
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        if cursor.fetchone():
            return
        try:
            cursor.execute('CREATE EXTENSION IF NOT EXISTS vector')
        except Exception as exc:
            if 'permission denied' in str(exc).lower() or 'rds_superuser' in str(exc).lower():
                raise RuntimeError(
                    'pgvector extension is not enabled on this database. '
                    'Connect to RDS as the master user and run:\n'
                    '  CREATE EXTENSION IF NOT EXISTS vector;\n'
                    'Then run migrate again.'
                ) from exc
            raise


def _drop_vector_extension(apps, schema_editor):
    with schema_editor.connection.cursor() as cursor:
        cursor.execute('DROP EXTENSION IF EXISTS vector')


class Migration(migrations.Migration):

    dependencies = [
        ('event', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(_ensure_vector_extension, _drop_vector_extension),

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
