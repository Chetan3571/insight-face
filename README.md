# InsightFace

A simple face recognition web application using InsightFace.

## Features

- Upload photos to albums with automatic face detection
- Search for similar faces across all uploaded photos
- Clean, simple interface

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run migrations:
   ```bash
   python manage.py migrate
   ```

3. Create sample data (optional):
   ```bash
   python manage.py create_sample_data
   ```

4. Run the development server:
   ```bash
   python manage.py runserver
   ```

## Usage

### Web Interface

- **Home**: View events and albums
- **Upload**: Select album and upload photos
- **Search**: Upload photo to find matching faces

### API Endpoints

- `GET /api/events/` - List events with albums
- `GET /api/albums/` - List albums with photos
- `POST /api/upload/` - Upload photos to album
- `POST /api/search/` - Search for faces

## Technologies

- Django
- InsightFace
- JavaScript
- `event/views.py`: API and frontend views
- `event/templates/`: HTML templates
- `event/face_utils.py`: Face recognition utilities
- `config/`: Django settings

## Technologies

- Django 6.0
- InsightFace
- PostgreSQL
- ONNX Runtime