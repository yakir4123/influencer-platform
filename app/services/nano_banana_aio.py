"""
NanoBananaAIO — Async version.
Supports Google Gemini Vertex AI (gemini-3-pro-image-preview / gemini-3.5-flash-image-preview)
using the asynchronous Google GenAI SDK.
"""

import os
import io
import logging
from PIL import Image
from typing import Optional, List, Union, Any
import google.auth
from google import genai
from google.genai import types

logger = logging.getLogger("app.services.nano_banana_aio")

_HARM_CATEGORIES = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
]


def _load_vertex_credentials(json_path: str):
    """Loads GCP service account credentials from a JSON key file."""
    from google.oauth2 import service_account

    try:
        creds = service_account.Credentials.from_service_account_file(json_path)
        project_id = creds.project_id
        return creds, project_id
    except Exception as e:
        raise RuntimeError(
            f"Failed to load service account credentials from {json_path}: {e}"
        )


def _load_vertex_json_folder(folder_path: str) -> List[str]:
    """Loads all JSON files in the specified folder path."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    json_files = sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(".json")
        ]
    )
    if not json_files:
        raise FileNotFoundError(f"No .json file found in: {folder_path}")
    return json_files


class NanoBananaAIO:
    _vertex_rotation_offset = 0

    async def generate_unified(
        self,
        provider: str = "VERTEX",
        prompt: str = "",
        negative_prompt: str = "",
        image_size: str = "1K",
        gemini_api_key: str = "",
        wavespeed_api_key: str = "",
        kie_api_key: str = "",
        fal_api_key: str = "",
        vertex_json_folder: str = "",
        disable_safety_threshold: bool = False,
        model: str = "Nano Banana Pro",
        batch_size: int = 1,
        use_search: bool = False,
        system_instructions: Optional[str] = None,
        aspect_ratio: str = "1:1",
        temperature: float = 1.0,
        top_p: float = 0.95,
        fal_safety_tolerance: str = "4",
        fal_enable_web_search: bool = False,
        image_1: Optional[Image.Image] = None,
        image_2: Optional[Image.Image] = None,
        video_mode_enabled: bool = False,
        face_swap_enabled: bool = False,
        breast_refiner_enabled: bool = False,
        low_neck_enabled: bool = False,
        face_expression: str = "Neutral",
        gpt2_image_quality: str = "high",
    ) -> tuple[list[Image.Image], str, str]:
        """
        Natively async generation method. Only VERTEX provider is supported.
        """
        if provider != "VERTEX":
            raise ValueError(
                f"Provider '{provider}' is not supported in this lightweight async runtime. Only 'VERTEX' is supported."
            )

        # Resolve credentials (rotation)
        vj_files = []
        if vertex_json_folder:
            try:
                vj_files = _load_vertex_json_folder(vertex_json_folder)
            except Exception as e:
                logger.error(f"Error reading vertex credentials folder: {e}")
            if vj_files:
                # Rotate credentials
                off = NanoBananaAIO._vertex_rotation_offset % len(vj_files)
                vj_files = vj_files[off:] + vj_files[:off]
                NanoBananaAIO._vertex_rotation_offset = (off + batch_size) % len(
                    vj_files
                )

        vertex_json_path = vj_files[0] if vj_files else ""
        credentials = None
        project_id = None

        if not vertex_json_path:
            try:
                credentials, project_id = google.auth.default()
                logger.info(f"Implicit ADC credentials loaded. Project: {project_id}")
            except Exception as e:
                raise RuntimeError(f"Failed to load implicit credentials: {e}")
        else:
            credentials, project_id = _load_vertex_credentials(vertex_json_path)
            logger.info(
                f"Loaded credentials from {vertex_json_path}. Project: {project_id}"
            )

        vertex_model_id = "gemini-2.5-flash-image"

        # Prepare HTTP options
        http_opts = genai.types.HttpOptions(timeout=300000)  # type: ignore

        client = genai.Client(
            vertexai=True,
            project=project_id,
            location="us-central1",
            credentials=credentials,
            http_options=http_opts,
        )

        # Prepare safety settings
        safety_threshold = "BLOCK_NONE" if disable_safety_threshold else None
        if safety_threshold:
            safety_settings = [
                types.SafetySetting(category=cat, threshold=safety_threshold)  # type: ignore
                for cat in _HARM_CATEGORIES
            ]
        else:
            safety_settings = None

        # Prepare GenerateContentConfig
        config_kwargs: dict[str, Any] = dict(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio, image_size=image_size
            ),
            temperature=temperature,
            top_p=top_p,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        if safety_settings:
            config_kwargs["safety_settings"] = safety_settings

        config = types.GenerateContentConfig(**config_kwargs)

        if system_instructions and system_instructions.strip():
            config.system_instruction = system_instructions

        if use_search:
            try:
                config.tools = [types.Tool(google_search=types.GoogleSearch())]
            except Exception as e:
                logger.warning(f"Failed to configure search tool: {e}")

        # Prepare contents
        contents: List[Union[str, Image.Image]] = [prompt]
        if image_1 is not None:
            # Convert to RGB to ensure compatibility (Gemini API does not support RGBA/alpha channel)
            img1_rgb = image_1.convert("RGB") if image_1.mode != "RGB" else image_1
            contents.append(img1_rgb)
        if image_2 is not None:
            # Convert to RGB to ensure compatibility (Gemini API does not support RGBA/alpha channel)
            img2_rgb = image_2.convert("RGB") if image_2.mode != "RGB" else image_2
            contents.append(img2_rgb)

        # Run async calls concurrently if batch_size > 1
        import asyncio

        async def _call_api():
            try:
                response = await client.aio.models.generate_content(
                    model=vertex_model_id,
                    contents=contents,
                    config=config,
                )

                # Check for blocked responses
                cands = getattr(response, "candidates", None) or []
                for cand in cands:
                    fr = getattr(cand, "finish_reason", None)
                    if (
                        fr
                        and fr not in (types.FinishReason.STOP,)
                        and str(fr) not in ("0", "FINISH_REASON_UNSPECIFIED", "None")
                    ):
                        raise RuntimeError(f"Generation blocked: finish_reason={fr}")

                # Find image bytes
                img_bytes = None
                text_resp = ""
                for cand in cands:
                    content = getattr(cand, "content", None)
                    if content and getattr(content, "parts", None):
                        for part in content.parts:
                            if getattr(part, "inline_data", None) and img_bytes is None:
                                img_bytes = part.inline_data.data
                            elif getattr(part, "text", None):
                                text_resp += part.text

                if not img_bytes:
                    raise RuntimeError("No image data in response candidates.")

                # Convert to PIL Image
                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                return pil_image, text_resp
            except Exception as e:
                logger.error(f"API call failed: {e}")
                raise e

        # Gather batch runs
        tasks = [_call_api() for _ in range(batch_size)]
        results = await asyncio.gather(*tasks)

        # Combine output images
        images = [r[0] for r in results]
        combined_text = "\n\n".join([r[1] for r in results])

        return images, combined_text, ""
