from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Influencer Platform API"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "local"
    DEBUG: bool = True

    # Instagram download settings
    INSTAGRAM_DOWNLOAD_TIMEOUT: int = 120
    INSTAGRAM_SAVE_TO: str = "temp"
    INSTAGRAM_COOKIES: Optional[str] = None

    # Google Cloud Storage settings
    GCS_BUCKET_NAME: Optional[str] = None

    # Google / Gemini AI settings
    GEMINI_API_KEY: Optional[str] = None

    # Telegram Bot Settings
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_BOT_WEBHOOK_URL: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


settings = Settings()
