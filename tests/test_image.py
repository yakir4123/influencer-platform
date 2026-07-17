import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PIL import Image
import io

from app.services.image import generate_fallback_image, generate_reposed_image
from app.schemas.image import ImageGenerationRequest


@pytest.fixture
def dummy_image():
    return Image.new("RGB", (100, 100), color="blue")


def test_generate_fallback_image_with_pil_image(dummy_image):
    out_path = generate_fallback_image(
        prompt="A photo of a person",
        preset="Normal",
        count_idx=0,
        image_2=dummy_image,
    )
    assert out_path.exists()
    assert out_path.suffix == ".png"
    # Verify we can open the resulting image and it's valid
    with Image.open(out_path) as img:
        assert img.size == (1024, 1024)
    # Clean up
    if out_path.exists():
        out_path.unlink()


def test_generate_fallback_image_with_local_path(tmp_path, dummy_image):
    local_img_path = tmp_path / "scene.png"
    dummy_image.save(local_img_path)

    out_path = generate_fallback_image(
        prompt="A photo of a person",
        preset="Normal",
        count_idx=0,
        image_2=str(local_img_path),
    )
    assert out_path.exists()
    # Verify we can open the resulting image and it's valid
    with Image.open(out_path) as img:
        assert img.size == (1024, 1024)
    # Clean up
    if out_path.exists():
        out_path.unlink()


@patch("app.services.image.httpx.get")
def test_generate_fallback_image_with_http_url(mock_httpx_get, dummy_image):
    img_byte_arr = io.BytesIO()
    dummy_image.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    mock_response = MagicMock()
    mock_response.content = img_bytes
    mock_httpx_get.return_value = mock_response

    out_path = generate_fallback_image(
        prompt="A photo of a person",
        preset="Normal",
        count_idx=0,
        image_2="http://example.com/scene.png",
    )
    assert out_path.exists()
    mock_httpx_get.assert_called_once_with("http://example.com/scene.png", timeout=10.0)
    # Clean up
    if out_path.exists():
        out_path.unlink()


@patch("app.services.gcs.download_gcs_file_bytes")
def test_generate_fallback_image_with_gcs_uri(mock_download_gcs, dummy_image):
    img_byte_arr = io.BytesIO()
    dummy_image.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    mock_download_gcs.return_value = img_bytes

    out_path = generate_fallback_image(
        prompt="A photo of a person",
        preset="Normal",
        count_idx=0,
        image_2="gs://my-bucket/posts/scene.webp",
    )
    assert out_path.exists()
    mock_download_gcs.assert_called_once_with("gs://my-bucket/posts/scene.webp")
    # Clean up
    if out_path.exists():
        out_path.unlink()


def test_generate_fallback_image_with_no_image():
    out_path = generate_fallback_image(
        prompt="A photo of a person",
        preset="Normal",
        count_idx=0,
        image_2=None,
    )
    assert out_path.exists()
    # Verify we can open the resulting image and it's valid
    with Image.open(out_path) as img:
        assert img.size == (1024, 1024)
    # Clean up
    if out_path.exists():
        out_path.unlink()


@pytest.mark.anyio
@patch("app.services.nano_banana_aio.NanoBananaAIO")
@patch("app.services.image.generate_fallback_image")
async def test_generate_reposed_image_fallback_mechanism(
    mock_generate_fallback, mock_nano_banana_aio_cls, tmp_path, dummy_image
):
    # Set up mocks
    mock_aio = MagicMock()
    mock_aio.generate_unified.side_effect = Exception("API offline")
    mock_nano_banana_aio_cls.return_value = mock_aio

    # Create dummy local reference image
    local_img_path = tmp_path / "scene.png"
    dummy_image.save(local_img_path)

    # Prepare fallback result mock
    mock_fallback_path = tmp_path / "fallback_result.png"
    dummy_image.save(mock_fallback_path)
    mock_generate_fallback.return_value = mock_fallback_path

    payload = ImageGenerationRequest(
        preset="Normal",
        prompt="re-pose prompt",
        count=1,
        retry_count=3,
        image_size="1K",
        image_2=str(local_img_path),
        identity_name="gal",
    )

    result = await generate_reposed_image(payload)

    assert result["status"] == "success"
    assert len(result["generated_images"]) == 1
    # Check that generate_fallback_image was called with the PIL Image object
    # (since the local image was successfully loaded as im2)
    mock_generate_fallback.assert_called_once()
    call_args = mock_generate_fallback.call_args[0]
    # Check arguments: (prompt, preset, count_idx, image_2)
    assert call_args[0] == "re-pose prompt"
    assert call_args[1] == "Normal"
    assert call_args[2] == 0
    # The 4th argument should be the loaded PIL Image object, not the string path
    assert isinstance(call_args[3], Image.Image)
