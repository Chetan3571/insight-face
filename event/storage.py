from django.conf import settings
from storages.backends.s3 import S3Storage


class R2MediaStorage(S3Storage):
    """Upload via the R2 S3 API; serve public URLs from R2_MEDIA_URL (r2.dev or custom domain)."""

    def url(self, name, parameters=None, expire=None, http_method=None):
        if settings.AWS_QUERYSTRING_AUTH:
            return super().url(name, parameters=parameters, expire=expire, http_method=http_method)
        base = settings.MEDIA_URL.rstrip('/')
        return f'{base}/{name}'
