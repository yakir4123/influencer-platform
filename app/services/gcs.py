import logging
from pathlib import Path
from typing import List, Optional
from google.cloud import storage
from app.core.config import settings

logger = logging.getLogger("app.services.gcs")


def get_gcs_client() -> Optional[storage.Client]:
    """
    Initializes and returns a GCS client if GCS_BUCKET_NAME is configured.
    """
    if not settings.GCS_BUCKET_NAME:
        return None
    try:
        return storage.Client()
    except Exception as e:
        logger.error(f"Failed to initialize GCS client: {e}")
        return None


def list_gcs_files(prefix: str) -> List[str]:
    """
    Lists all files (GCS URIs) under a given prefix in the configured bucket.
    """
    client = get_gcs_client()
    if not client or not settings.GCS_BUCKET_NAME:
        return []
    
    try:
        bucket = client.bucket(settings.GCS_BUCKET_NAME)
        # Ensure prefix has a trailing slash for distinct folders
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        blobs = bucket.list_blobs(prefix=prefix)
        # We only want actual files, not prefix directories (blobs ending with /)
        return sorted([
            f"gs://{settings.GCS_BUCKET_NAME}/{blob.name}"
            for blob in blobs
            if not blob.name.endswith("/")
        ])
    except Exception as e:
        logger.error(f"Failed to list GCS files for prefix {prefix}: {e}")
        return []


def upload_file_to_gcs(local_path: Path, gcs_path: str) -> Optional[str]:
    """
    Uploads a local file to GCS and returns the GCS URI.
    """
    client = get_gcs_client()
    if not client or not settings.GCS_BUCKET_NAME:
        return None
    
    try:
        bucket = client.bucket(settings.GCS_BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_path))
        return f"gs://{settings.GCS_BUCKET_NAME}/{gcs_path}"
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to GCS path {gcs_path}: {e}")
        return None


def delete_gcs_files(prefix: str) -> None:
    """
    Deletes all files under a prefix in the configured GCS bucket.
    """
    client = get_gcs_client()
    if not client or not settings.GCS_BUCKET_NAME:
        return
    
    try:
        bucket = client.bucket(settings.GCS_BUCKET_NAME)
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        blobs = list(bucket.list_blobs(prefix=prefix))
        if blobs:
            bucket.delete_blobs(blobs)
            logger.info(f"Deleted {len(blobs)} existing blobs in GCS prefix: {prefix}")
    except Exception as e:
        logger.error(f"Failed to delete GCS files for prefix {prefix}: {e}")


def download_gcs_file_bytes(gcs_uri: str) -> Optional[bytes]:
    """
    Downloads and returns the bytes of a GCS object specified by a gs:// URI.
    """
    if not gcs_uri.startswith("gs://"):
        return None
    client = get_gcs_client()
    if not client:
        return None
    try:
        parts = gcs_uri[5:].split("/", 1)
        if len(parts) != 2:
            return None
        bucket_name, blob_name = parts
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_bytes()
    except Exception as e:
        logger.error(f"Failed to download GCS file {gcs_uri}: {e}")
        return None

