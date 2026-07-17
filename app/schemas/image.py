from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ImageGenerationRequest(BaseModel):
    preset: Literal["Subtle", "Normal", "Heavy"] = Field(
        ...,
        description="Controls how much the model is allowed to deviate from the original",
    )
    prompt: str = Field(
        ..., description="Describe the subject, style, or any specific detail"
    )
    count: int = Field(
        ...,
        ge=1,
        le=20,
        description="Number of re-posed images to generate per scene image",
    )
    retry_count: int = Field(
        ..., ge=1, le=10, description="Number of retries per image on failure"
    )
    image_size: Literal["1K", "2K", "4K", "8K"] = Field(
        ..., description="Output resolution"
    )

    # Optional parameters
    image_2: Optional[str] = Field(
        default=None, description="Scene/environment reference image (URL or base64)"
    )
    is_selfie: bool = Field(
        default=False, description="Adds selfie framing instructions to the prompt"
    )
    is_mirror_selfie: bool = Field(
        default=False, description="Adds mirror selfie instructions to the prompt"
    )
    vertex_json_folder: str = Field(
        default="", description="Path to service account JSON folder"
    )
    disable_safety_threshold: bool = Field(
        default=False, description="Disable the safety filter"
    )
    aspect_ratio: Literal[
        "auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"
    ] = Field(default="auto", description="Aspect ratio")
    temperature: float = Field(
        default=1.0, ge=0.0, le=2.0, description="Model creativity"
    )
    identity_name: Optional[str] = Field(
        default="gal", description="Identity reference name (e.g. gal)"
    )


class ImageGenerationResponse(BaseModel):
    status: str = "success"
    generated_images: List[str]
    message: str
