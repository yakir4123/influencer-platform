from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.health import HealthStatus
from app.schemas.instagram import InstagramDownloadRequest, InstagramDownloadResponse
from app.schemas.image import ImageGenerationRequest, ImageGenerationResponse
from app.services.instagram import download_instagram_post

router = APIRouter()


@router.get("/health", response_model=HealthStatus)
def health_check() -> HealthStatus:
    """
    Health check endpoint to verify the server is running and configuration is correct.
    """
    return HealthStatus(status="ok", environment=settings.ENVIRONMENT)


@router.post("/download-post", response_model=InstagramDownloadResponse)
def download_post(payload: InstagramDownloadRequest) -> InstagramDownloadResponse:
    """
    Download an entire Instagram post from a given URL.
    """
    try:
        result = download_instagram_post(
            source_url=payload.url,
            max_items=payload.max_items,
            force_refresh=payload.force_refresh,
        )
        return InstagramDownloadResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download Instagram post: {str(e)}"
        )


@router.post("/generate-image", response_model=ImageGenerationResponse)
def generate_image(payload: ImageGenerationRequest) -> ImageGenerationResponse:
    """
    Generate images based on a text prompt using Vertex AI (mocked).
    """
    return ImageGenerationResponse(
        status="success",
        generated_images=["mock_image_url_1.png"],
        message="works!"
    )

