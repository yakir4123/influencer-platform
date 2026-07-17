from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.core.config import settings
from app.bot import run_bot_async, stop_bot_async


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.TELEGRAM_BOT_TOKEN:
        await run_bot_async()
    yield
    await stop_bot_async()


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend API for Influencer Platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the API router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
def read_root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}!",
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/health",
    }
