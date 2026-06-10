"""
Django settings for config project.

All environment-specific values are read from .env (see .env.example).
Copy .env.example to .env and edit only that file for local or staging.
"""

import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')


def _env(key: str, default: str | None = None) -> str | None:
    value = os.environ.get(key)
    if value is None or value == '':
        return default
    return value


def _env_bool(key: str, *, default: bool) -> bool:
    raw = _env(key)
    if raw is None:
        return default
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


def _env_list(key: str, *, default: list[str]) -> list[str]:
    raw = _env(key)
    if not raw:
        return default
    return [item.strip() for item in raw.split(',') if item.strip()]


DEBUG = _env_bool('DJANGO_DEBUG', default=True)

_secret_key = _env('DJANGO_SECRET_KEY')
if _secret_key:
    SECRET_KEY = _secret_key
elif DEBUG:
    SECRET_KEY = 'django-insecure-dev-only-key-set-DJANGO_SECRET_KEY-in-env'
else:
    raise ImproperlyConfigured('Set DJANGO_SECRET_KEY in .env when DJANGO_DEBUG=False')

ALLOWED_HOSTS = _env_list('DJANGO_ALLOWED_HOSTS', default=['*'])

# Public API — no session auth (avoids CSRF token requirement on POST uploads)
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
}


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'django_celery_beat',
    'django_celery_results',
    'pgvector.django',
    'django.contrib.postgres',
    'event',
]

MIDDLEWARE = [
    'event.middleware.CORSMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# CORS Settings for testing
CORS_ALLOWED_ORIGINS = ['*']
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True
CORS_EXPOSE_HEADERS = '*'

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'


# Database — values from .env (DB_PORT 5433 matches docker-compose host mapping)

_db_password = _env('DB_PASSWORD')
if not _db_password and not DEBUG:
    raise ImproperlyConfigured('Set DB_PASSWORD in .env when DJANGO_DEBUG=False')

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': _env('DB_NAME', 'insight_face'),
        'USER': _env('DB_USER', 'chetanface'),
        'PASSWORD': _db_password or '',
        'HOST': _env('DB_HOST', 'localhost'),
        'PORT': _env('DB_PORT', '5433'),
    }
}


# Password validation
# https://docs.djangoproject.com/en/6.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/6.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static / media — paths are under project root (e.g. .../insight-face/staticfiles on staging)
# https://docs.djangoproject.com/en/6.0/howto/static-files/

STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DATA_UPLOAD_MAX_MEMORY_SIZE = 104857600   # 100 MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 104857600  # 100 MB

DATA_UPLOAD_MAX_NUMBER_FILES = 1000

CELERY_BROKER_URL = _env('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = 'django-db'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_EXTENDED = True

# Logging — file + console on staging; console only when DEBUG=True (local)
_LOG_FORMATTER = {
    'format': '{asctime} [{levelname}] {name}: {message}',
    'style': '{',
}

if DEBUG:
    LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {'verbose': _LOG_FORMATTER},
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'verbose',
            },
        },
        'loggers': {
            'django': {'handlers': ['console'], 'level': 'INFO'},
            'event': {
                'handlers': ['console'],
                'level': 'INFO',
                'propagate': False,
            },
        },
    }
else:
    LOG_DIR = BASE_DIR / 'logs'
    LOG_DIR.mkdir(exist_ok=True)

    LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {'verbose': _LOG_FORMATTER},
        'handlers': {
            'file': {
                'level': 'INFO',
                'class': 'logging.FileHandler',
                'filename': LOG_DIR / 'django.log',
                'formatter': 'verbose',
            },
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'verbose',
            },
        },
        'loggers': {
            'django': {
                'handlers': ['file', 'console'],
                'level': 'INFO',
            },
            'event': {
                'handlers': ['file', 'console'],
                'level': 'INFO',
                'propagate': False,
            },
        },
    }
