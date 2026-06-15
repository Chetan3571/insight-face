import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def local_image_paths(image_fields):
    """
    Yield local filesystem paths for ImageField files.

    Uses the on-disk path for local storage; downloads to temp files when the
    default backend is remote (e.g. Cloudflare R2).
    """
    fields = list(image_fields)
    temp_files: list[str] = []
    paths: list[str] = []

    try:
        for field in fields:
            try:
                paths.append(field.path)
            except (NotImplementedError, ValueError):
                suffix = Path(field.name).suffix or '.jpg'
                with field.open('rb') as src:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(src.read())
                        temp_path = tmp.name
                temp_files.append(temp_path)
                paths.append(temp_path)
        yield paths
    finally:
        for temp_path in temp_files:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
