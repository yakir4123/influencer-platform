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
