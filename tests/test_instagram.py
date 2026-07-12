from unittest.mock import patch
import pytest


def test_download_post_endpoint_success(client):
    mock_response = {
        "original_url": "https://www.instagram.com/p/C_abc123XYZ/?img_index=1",
        "download_url": "https://www.instagram.com/p/C_abc123XYZ",
        "selected_image_index": 0,
        "url_img_index_override": 0,
        "download_dir": "/absolute/path/to/downloads/social_downloads/instagram.com_p_C_abc123XYZ_abc123",
        "media_files": [
            "/absolute/path/to/downloads/social_downloads/instagram.com_p_C_abc123XYZ_abc123/img1.jpg",
            "/absolute/path/to/downloads/social_downloads/instagram.com_p_C_abc123XYZ_abc123/img2.jpg"
        ],
        "summary": {
            "mode": "gallery-dl",
            "order_source": "gallery-dl stdout order",
            "downloaded_count": 2,
            "required_min_count": 1,
            "out_dir": "/absolute/path/to/downloads/social_downloads/instagram.com_p_C_abc123XYZ_abc123",
            "returncode": 0,
            "elapsed_seconds": 1.25,
            "files": [
                {"index": 0, "name": "img1.jpg", "path": "/absolute/path/to/downloads/social_downloads/instagram.com_p_C_abc123XYZ_abc123/img1.jpg"},
                {"index": 1, "name": "img2.jpg", "path": "/absolute/path/to/downloads/social_downloads/instagram.com_p_C_abc123XYZ_abc123/img2.jpg"}
            ],
            "single_instagram_post": True,
            "cookies_used": False,
            "range_applied": False,
            "max_items": 10
        }
    }

    with patch("app.api.routes.download_instagram_post", return_value=mock_response) as mock_download:
        payload = {
            "url": "https://www.instagram.com/p/C_abc123XYZ/?img_index=1",
            "max_items": 10,
            "force_refresh": True
        }
        response = client.post("/api/v1/download-post", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["original_url"] == payload["url"]
        assert len(data["media_files"]) == 2
        mock_download.assert_called_once_with(
            source_url=payload["url"],
            max_items=payload["max_items"],
            force_refresh=payload["force_refresh"]
        )


def test_download_post_endpoint_failure(client):
    with patch("app.api.routes.download_instagram_post", side_effect=Exception("gallery-dl crashed")):
        payload = {
            "url": "https://www.instagram.com/p/C_abc123XYZ/",
        }
        response = client.post("/api/v1/download-post", json=payload)
        assert response.status_code == 400
        assert "gallery-dl crashed" in response.json()["detail"]


import json
from pathlib import Path
from PIL import Image
from unittest.mock import patch
from app.services.image import compress_image_lossless
from app.services.instagram import download_instagram_post


def test_compress_image_lossless(tmp_path):
    # Create a dummy JPEG image
    img_path = tmp_path / "test_image.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path, "JPEG")
    
    assert img_path.exists()
    
    # Compress it
    webp_path = compress_image_lossless(img_path)
    
    assert webp_path.suffix == ".webp"
    assert webp_path.exists()
    assert not img_path.exists()  # Original should be deleted
    
    # Check that it is still a valid image
    with Image.open(webp_path) as opened_img:
        assert opened_img.size == (100, 100)


@patch("app.services.instagram.list_gcs_files")
@patch("app.services.instagram.settings")
def test_instagram_download_gcs_cache_hit(mock_settings, mock_list_gcs):
    mock_settings.GCS_BUCKET_NAME = "test-bucket"
    mock_settings.INSTAGRAM_DOWNLOAD_TIMEOUT = 120
    mock_settings.INSTAGRAM_SAVE_TO = "temp"
    mock_settings.INSTAGRAM_COOKIES = None
    
    # Mock cached GCS files
    mock_list_gcs.return_value = [
        "gs://test-bucket/social_downloads/www.instagram.com_p_C_abc123XYZ_eadf2f7ab3/img1.webp",
        "gs://test-bucket/social_downloads/www.instagram.com_p_C_abc123XYZ_eadf2f7ab3/img2.webp"
    ]
    
    result = download_instagram_post(
        source_url="https://www.instagram.com/p/C_abc123XYZ/?img_index=1",
        max_items=10,
        force_refresh=False
    )
    
    assert result["download_dir"] == "gs://test-bucket/social_downloads/www.instagram.com_p_C_abc123XYZ_eadf2f7ab3"
    assert len(result["media_files"]) == 2
    assert result["media_files"][0] == "gs://test-bucket/social_downloads/www.instagram.com_p_C_abc123XYZ_eadf2f7ab3/img1.webp"
    assert result["summary"]["mode"] == "gcs-cache"
    mock_list_gcs.assert_called_once()


@patch("app.services.instagram.upload_file_to_gcs")
@patch("app.services.instagram.list_gcs_files")
@patch("app.services.instagram.download_with_gallery_dl")
@patch("app.services.instagram.settings")
def test_instagram_download_gcs_cache_miss(mock_settings, mock_gallery_dl, mock_list_gcs, mock_upload):
    mock_settings.GCS_BUCKET_NAME = "test-bucket"
    mock_settings.INSTAGRAM_DOWNLOAD_TIMEOUT = 120
    mock_settings.INSTAGRAM_SAVE_TO = "temp"
    mock_settings.INSTAGRAM_COOKIES = None
    
    # Cache miss
    mock_list_gcs.return_value = []
    
    # Mock gallery-dl downloads
    def fake_download_with_gallery_dl(url, out_dir, cookies_file, max_items, timeout, selected_image_index):
        # Create a fake file in the output directory
        p = Path(out_dir) / "img1.jpg"
        img = Image.new("RGB", (50, 50), color="blue")
        img.save(p, "JPEG")
        summary = {
            "mode": "gallery-dl",
            "downloaded_count": 1,
            "out_dir": str(out_dir),
            "files": [{"index": 0, "name": "img1.jpg", "path": str(p)}]
        }
        return [str(p)], json.dumps(summary)
        
    mock_gallery_dl.side_effect = fake_download_with_gallery_dl
    
    # Mock upload
    mock_upload.side_effect = lambda local_file, gcs_path: f"gs://test-bucket/{gcs_path}"
    
    result = download_instagram_post(
        source_url="https://www.instagram.com/p/C_abc123XYZ/",
        max_items=10,
        force_refresh=False
    )
    
    assert result["download_dir"] == "gs://test-bucket/social_downloads/www.instagram.com_p_C_abc123XYZ_eadf2f7ab3"
    assert len(result["media_files"]) == 1
    assert result["media_files"][0] == "gs://test-bucket/social_downloads/www.instagram.com_p_C_abc123XYZ_eadf2f7ab3/img1.webp"
    assert result["summary"]["mode"] == "gcs-gallery-dl"
    mock_upload.assert_called_once()
