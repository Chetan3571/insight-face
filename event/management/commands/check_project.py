import os
import subprocess
import sys
from pathlib import Path

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.db import connection


class Command(BaseCommand):
    help = 'Verify database, pgvector, Redis, Docker services, and optional InsightFace setup.'
    requires_system_checks = []

    def add_arguments(self, parser):
        parser.add_argument(
            '--full',
            action='store_true',
            help='Also load InsightFace models (slow, ~30-60s on CPU).',
        )
        parser.add_argument(
            '--skip-docker',
            action='store_true',
            help='Skip Docker container status checks.',
        )

    def handle(self, *args, **options):
        self.ok = 0
        self.warn = 0
        self.fail = 0
        self.full = options['full']
        self.skip_docker = options['skip_docker']

        self.stdout.write(self.style.MIGRATE_HEADING('insight-face project check\n'))

        self._check_env()
        self._check_packages()
        self._check_django()
        self._check_database()
        self._check_pgvector()
        self._check_migrations()
        self._check_redis()
        self._check_redis_cache()
        self._check_media()
        if not self.skip_docker:
            self._check_docker()
        self._check_celery()
        if self.full:
            self._check_insightface()

        self.stdout.write('')
        self.stdout.write(
            f'Summary: {self.ok} passed, {self.warn} warnings, {self.fail} failed'
        )

        if self.fail:
            sys.exit(1)

    def _pass(self, label, detail=''):
        self.ok += 1
        msg = f'  OK   {label}'
        if detail:
            msg += f' — {detail}'
        self.stdout.write(self.style.SUCCESS(msg))

    def _warn(self, label, detail=''):
        self.warn += 1
        msg = f'  WARN {label}'
        if detail:
            msg += f' — {detail}'
        self.stdout.write(self.style.WARNING(msg))

    def _fail(self, label, detail=''):
        self.fail += 1
        msg = f'  FAIL {label}'
        if detail:
            msg += f' — {detail}'
        self.stdout.write(self.style.ERROR(msg))

    def _check_env(self):
        self.stdout.write('Environment')
        env_file = settings.BASE_DIR / '.env'
        if env_file.exists():
            self._pass('.env file', str(env_file))
        else:
            self._warn('.env file', 'not found — using defaults / shell env')

        if settings.DEBUG:
            self._warn('DEBUG', 'True (disable on staging/production)')
        else:
            self._pass('DEBUG', 'False')

        hosts = list(settings.ALLOWED_HOSTS)
        if not hosts or hosts == ['*']:
            self._warn('ALLOWED_HOSTS', repr(hosts) or 'empty')
        else:
            self._pass('ALLOWED_HOSTS', ', '.join(hosts))

    def _check_packages(self):
        self.stdout.write('\nPython packages')
        required = ('django', 'celery', 'redis', 'psycopg', 'pgvector', 'insightface', 'cv2', 'gdown')
        for name in required:
            try:
                if name == 'cv2':
                    __import__('cv2')
                elif name == 'psycopg':
                    __import__('psycopg')
                else:
                    __import__(name)
                self._pass(name)
            except ImportError as exc:
                self._fail(name, str(exc))

    def _check_django(self):
        self.stdout.write('\nDjango')
        try:
            call_command(
                'check',
                verbosity=0,
                tags=['models', 'database', 'files', 'caches', 'templates'],
            )
            self._pass('system check')
        except SystemExit:
            self._fail('system check', 'django check reported errors')

    def _check_database(self):
        self.stdout.write('\nDatabase')
        db = settings.DATABASES['default']
        self._pass(
            'config',
            f"{db['USER']}@{db['HOST']}:{db['PORT']}/{db['NAME']}",
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute('SELECT 1')
            self._pass('connection')
        except Exception as exc:
            self._fail('connection', str(exc))

    def _check_pgvector(self):
        self.stdout.write('\npgvector')
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
                )
                if cursor.fetchone():
                    self._pass('extension installed')
                else:
                    self._fail(
                        'extension installed',
                        'run: CREATE EXTENSION vector; (or use pgvector Docker image)',
                    )

                cursor.execute(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_name = 'event_faceembedding'"
                )
                if cursor.fetchone():
                    self._pass('event_faceembedding table')
                else:
                    self._fail('event_faceembedding table', 'run: python manage.py migrate')
        except Exception as exc:
            self._fail('pgvector', str(exc))

    def _check_migrations(self):
        self.stdout.write('\nMigrations')
        try:
            from io import StringIO

            out = StringIO()
            call_command('showmigrations', '--plan', stdout=out, no_color=True)
            plan = out.getvalue()
            unapplied = [
                line for line in plan.splitlines()
                if line.strip().startswith('[ ]')
            ]
            if unapplied:
                self._fail('pending migrations', f'{len(unapplied)} unapplied')
            else:
                self._pass('all migrations applied')
        except Exception as exc:
            self._fail('migration check', str(exc))

    def _check_redis(self):
        self.stdout.write('\nRedis / Celery broker')
        broker = (
            getattr(settings, 'CELERY_BROKER_URL', None)
            or os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
        )
        self._pass('broker URL', broker)

        try:
            import redis
        except ImportError:
            self._fail('redis package', 'pip install redis')
            return

        try:
            client = redis.from_url(broker, socket_connect_timeout=3)
            if client.ping():
                self._pass('redis ping')
            else:
                self._fail('redis ping', 'no response')
        except Exception as exc:
            self._fail('redis ping', str(exc))

    def _check_redis_cache(self):
        self.stdout.write('\nRedis cache (search results)')
        cache_url = getattr(settings, 'CACHES', {}).get('default', {}).get('LOCATION', '')
        self._pass('cache URL', str(cache_url))

        try:
            from django.core.cache import cache

            cache.set('_healthcheck', 'ok', 10)
            if cache.get('_healthcheck') == 'ok':
                self._pass('cache read/write')
            else:
                self._warn('cache read/write', 'miss — Redis DB 1 down or IGNORE_EXCEPTIONS')
        except Exception as exc:
            self._warn('cache read/write', str(exc))

    def _check_media(self):
        self.stdout.write('\nMedia storage')
        if getattr(settings, 'USE_R2_STORAGE', False):
            from django.core.files.storage import default_storage

            bucket = getattr(settings, 'AWS_STORAGE_BUCKET_NAME', '?')
            test_name = '.write_test'
            try:
                default_storage.save(test_name, ContentFile(b'ok'))
                default_storage.delete(test_name)
                self._pass('R2 storage writable', bucket)
            except Exception as exc:
                self._fail('R2 storage writable', str(exc))
            return

        media_root = Path(settings.MEDIA_ROOT)
        try:
            media_root.mkdir(parents=True, exist_ok=True)
            test_file = media_root / '.write_test'
            test_file.write_text('ok')
            test_file.unlink()
            self._pass('MEDIA_ROOT writable', str(media_root))
        except Exception as exc:
            self._fail('MEDIA_ROOT writable', str(exc))

    def _check_docker(self):
        self.stdout.write('\nDocker')
        compose_file = settings.BASE_DIR / 'docker-compose.yml'
        if not compose_file.exists():
            self._warn('docker-compose.yml', 'not found — skipped')
            return

        docker_sh = settings.BASE_DIR / 'deploy' / 'docker.sh'
        cmd = [str(docker_sh), 'ps', '--format', 'json'] if docker_sh.exists() else ['docker', 'compose', 'ps', '--format', 'json']

        try:
            result = subprocess.run(
                cmd,
                cwd=settings.BASE_DIR,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except FileNotFoundError:
            self._warn('docker', 'not installed — skipped')
            return
        except subprocess.TimeoutExpired:
            self._fail('docker', 'timed out')
            return

        if result.returncode != 0:
            self._warn(
                'docker compose ps',
                (result.stderr or result.stdout or 'failed').strip()[:120],
            )
            return

        output = result.stdout.lower()
        for service in ('db', 'redis'):
            if service in output and 'running' in output:
                self._pass(f'{service} container')
            else:
                self._warn(
                    f'{service} container',
                    f'not running — try: ./deploy/docker.sh up -d {service}',
                )

    def _check_celery(self):
        self.stdout.write('\nCelery worker')
        try:
            from config.celery import app as celery_app

            inspector = celery_app.control.inspect(timeout=3)
            ping = inspector.ping()
            if ping:
                workers = ', '.join(ping.keys())
                self._pass('worker reachable', workers)
            else:
                self._warn(
                    'worker reachable',
                    'no workers — start: celery -A config worker -l info',
                )
        except Exception as exc:
            self._warn('worker reachable', str(exc))

    def _check_insightface(self):
        self.stdout.write('\nAdaFace pipeline')
        pack_dir = Path.home() / '.insightface' / 'models' / 'adaface'
        required = (
            'det_10g.onnx',
            '2d106det.onnx',
            'adaface_ir101_webface12m.onnx',
        )
        for name in required:
            path = pack_dir / name
            if path.exists() and path.stat().st_size > 100_000:
                self._pass(name, str(path))
            else:
                self._warn(name, 'will download on first Celery task')

        try:
            from event import face_utils
            recog = face_utils.app.models.get('recognition')
            model_name = getattr(recog, 'model_file', 'unknown')
            self._pass('face_utils import', f'recognition: {model_name}')
        except Exception as exc:
            self._fail('face_utils import', str(exc))
