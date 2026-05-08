from django.http import HttpResponse
from django.conf import settings


class CORSMiddleware:
    """Add permissive CORS headers and handle OPTIONS preflight requests."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.method == 'OPTIONS':
            response = HttpResponse()
        else:
            response = self.get_response(request)

        self._add_cors_headers(response)
        return response

    def _add_cors_headers(self, response):
        response.setdefault('Access-Control-Allow-Origin', '*')
        response.setdefault('Access-Control-Allow-Methods', 'GET, POST, PUT, PATCH, DELETE, OPTIONS')
        response.setdefault(
            'Access-Control-Allow-Headers',
            'Content-Type, Authorization, X-Requested-With, X-CSRFToken'
        )
        response.setdefault('Access-Control-Allow-Credentials', 'true')
        response.setdefault('Access-Control-Expose-Headers', 'Content-Type, Authorization')
        return response
