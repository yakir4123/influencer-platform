from typing import List, Optional
from pydantic import BaseModel


class ImageGenerationRequest(BaseModel):
    prompt: str
    number_of_images: int = 1
    aspect_ratio: str = "1:1"


class ImageGenerationResponse(BaseModel):
    status: str = "success"
    generated_images: List[str]
    message: str
