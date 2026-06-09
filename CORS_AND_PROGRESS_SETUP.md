# Upload Progress Tracker & CORS Configuration

## Changes Made

### 1. CORS Configuration (settings.py)
- **Allowed Hosts**: Set to '*' (allow all hosts)
- **Removed CSRF Middleware**: Removed `django.middleware.csrf.CsrfViewMiddleware` for unrestricted API access
- **CORS Settings**: Added configuration to allow cross-origin requests from anywhere

### 2. Custom CORS Middleware (middleware.py)
- Created `event/middleware.py` with `CORSMiddleware` class
- Adds CORS headers to all responses:
  - `Access-Control-Allow-Origin: *`
  - `Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS`
  - `Access-Control-Allow-Headers: Content-Type, Authorization`

### 3. CORS Decorators (views.py)
- Added `add_cors_headers()` decorator to all API endpoints
- Handles OPTIONS requests for CORS preflight
- Applied to:
  - `upload_photo()` - File upload endpoint
  - `search_by_face()` - Face search endpoint
  - `list_events()` - Events API
  - `list_albums()` - Albums API

### 4. Upload Progress Tracker (upload.html)
- Real-time progress bar showing upload percentage
- Displays uploaded/total file size in MB
- Uses XMLHttpRequest with upload progress events
- Progress visualization:
  - Progress bar with percentage indicator
  - File size information
  - Completion message

## Features

✅ **No CORS Errors**: All requests allowed from any origin
✅ **No Authentication**: Open access for testing
✅ **No CSRF Protection**: Disabled for API endpoints
✅ **Progress Tracking**: Real-time upload progress with visual feedback
✅ **File Size Display**: Shows upload progress in MB
✅ **OPTIONS Support**: Handles CORS preflight requests

## API Access

All endpoints accessible from anywhere:
```
GET  /api/events/     - List all events
GET  /api/albums/     - List all albums
POST /api/upload/     - Upload photos
POST /api/search/     - Search faces
```

## Server Status

- Server running on: **http://127.0.0.1:8004/**
- All CORS errors removed
- No authentication required
- No security restrictions (testing stage)
