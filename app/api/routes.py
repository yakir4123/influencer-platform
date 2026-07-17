from fastapi import APIRouter, HTTPException, Request

from app.core.config import settings
from app.schemas.health import HealthStatus
from app.schemas.instagram import InstagramDownloadRequest, InstagramDownloadResponse
from app.schemas.image import ImageGenerationRequest, ImageGenerationResponse
from app.services.instagram import download_instagram_post
from app.services.image import generate_reposed_image

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
    try:
        result = generate_reposed_image(payload)
        return ImageGenerationResponse(**result)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to generate image: {str(e)}"
        )


@router.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    """
    Webhook endpoint for receiving Telegram updates in production.
    """
    import hashlib
    from app.bot import get_bot_app
    from telegram import Update

    if not settings.TELEGRAM_BOT_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="Telegram bot token is not configured on the server."
        )

    # Validate secret token header
    header_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    expected_token = hashlib.sha256(settings.TELEGRAM_BOT_TOKEN.encode()).hexdigest()
    
    if header_token != expected_token:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Invalid webhook secret token."
        )

    bot_app = get_bot_app()
    if not bot_app:
        raise HTTPException(
            status_code=503,
            detail="Telegram bot application is not currently initialized."
        )

    try:
        update_data = await request.json()
        update = Update.de_json(update_data, bot_app.bot)
        
        # Schedule update handling in background task to respond to Telegram immediately
        import asyncio
        asyncio.create_task(bot_app.process_update(update))
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process Telegram update: {str(e)}"
        )




