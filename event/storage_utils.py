import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path


def _download_field(field) -> tuple[str, str | None]:
    """Return (local_path, temp_path_or_None) for one ImageField.

    Local storage fields resolve instantly via field.path.
    Remote fields (R2) are downloaded to a temp file.
    """
    try:
        return field.path, None
    except (NotImplementedError, ValueError):
        pass
    suffix = Path(field.name).suffix or '.jpg'
    with field.open('rb') as src:
        data = src.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.flush()
        return tmp.name, tmp.name
    finally:
        tmp.close()


@contextmanager
def local_image_paths(image_fields, max_workers: int | None = None):
    """
    Yield local filesystem paths for ImageField files.

    Local storage: resolves field.path directly (no I/O).
    Remote storage (e.g. Cloudflare R2): downloads all files in parallel via
    ThreadPoolExecutor, then cleans up temp files on exit.

    max_workers defaults to R2_DOWNLOAD_THREADS env var (default 4).
    """
    fields = list(image_fields)
    if not fields:
        yield []
        return

    workers = max_workers or int(os.environ.get('R2_DOWNLOAD_THREADS', '4'))
    temp_files: list[str] = []
    paths: list[str] = [None] * len(fields)  # type: ignore[list-item]

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {pool.submit(_download_field, f): i for i, f in enumerate(fields)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                local_path, temp_path = future.result()
                paths[idx] = local_path
                if temp_path:
                    temp_files.append(temp_path)
        yield paths
    finally:
        for temp_path in temp_files:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
