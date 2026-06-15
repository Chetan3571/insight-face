"""
Delete photos from Cloudflare R2 and the database.

  pip install boto3

Edit CONFIG, then from project root:

  exec(open("delete_r2_photos.py").read())

Or: python manage.py shell
"""

from __future__ import annotations
import boto3
import django
from botocore.config import Config

from event.models import Photo

# --- CONFIG ---
ACCOUNT_ID = "4e1e7927e4cd7c80678710261c15bf1f"
ACCESS_KEY_ID = "913bef25451804f80d36cf411788ace5"
SECRET_ACCESS_KEY = "82707e552d5d227308acabf2ec1e7ad4e0e34845c6ed9d03e93f352eeeb4a0ce"
BUCKET_NAME = "snapdme-testing"
PATH_PREFIX = "photos/"
# --------------


def _client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def _list_keys(client, prefix):
    keys = []
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET_NAME, Prefix=prefix
    ):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def _delete_keys(client, keys):
    deleted = 0
    for i in range(0, len(keys), 1000):
        batch = keys[i : i + 1000]
        resp = client.delete_objects(
            Bucket=BUCKET_NAME,
            Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
        )
        deleted += len(resp.get("Deleted", []))
        for err in resp.get("Errors", []):
            print(f"R2 FAILED {err.get('Key')}: {err.get('Message')}")
    return deleted


if "your-" in ACCOUNT_ID or "your-" in ACCESS_KEY_ID:
    raise ValueError("Edit CONFIG at the top of delete_r2_photos.py first.")

photos = Photo.objects.filter(image__startswith=PATH_PREFIX)
photo_count = photos.count()
print(f"DB: {photo_count} photo(s) matching '{PATH_PREFIX}'")
for photo in photos:
    print(f"  id={photo.id} {photo.image.name}")
    photo.image.delete(save=False)
db_deleted, _ = photos.delete()
print(f"DB: deleted {db_deleted} row(s) (includes face embeddings)")

client = _client()
keys = _list_keys(client, PATH_PREFIX)

if not keys:
    print(f"R2: no objects under s3://{BUCKET_NAME}/{PATH_PREFIX}")
else:
    print(f"R2: {len(keys)} object(s) under s3://{BUCKET_NAME}/{PATH_PREFIX}")
    for key in keys:
        print(f"  {key}")
    r2_deleted = _delete_keys(client, keys)
    print(f"R2: deleted {r2_deleted} object(s).")
