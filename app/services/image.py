import asyncio
import io
import logging
import random
import time
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw, ImageFilter
import aiohttp

logger = logging.getLogger("app.services.image")


def compress_image_lossless(file_path: Path) -> Path:
    """
    Losslessly compresses an image.
    Converts standard image formats (JPEG, PNG, BMP, TIFF) to WebP lossless.
    Leaves other file types (like videos or already WebP files) as-is.
    """
    suffix = file_path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
        return file_path

    # Convert to WebP lossless
    output_path = file_path.with_suffix(".webp")
    try:
        with Image.open(file_path) as img:
            # webp format supports lossless compression
            img.save(output_path, format="WEBP", lossless=True, quality=100)
        
        # If output_path is different from the original file, delete the original
        if output_path != file_path and file_path.exists():
            file_path.unlink()
        logger.info(f"Losslessly compressed {file_path.name} -> {output_path.name}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to losslessly compress {file_path}: {e}")
        return file_path


_PRESET_DIRECTIVES = {
    "Subtle": (
        "Picture 1 defines the identity, facial anatomy, facial proportions, skin texture and hair characteristics of the subject.\n"
        "Picture 2 defines the environment, outfit, camera framing, pose, perspective and overall scene composition.\n"
        "Generate one photorealistic image of the exact same person from picture 1 naturally existing inside the scene of picture 2.\n"
        "The subject must preserve the exact facial likeness from picture 1:\n"
        "same eye shape, eye spacing, eyelids, nose shape, lips, jawline, cheek structure, facial proportions, skin tone and hair characteristics.\n"
        "Maintain extremely high identity consistency with picture 1.\n"
        "Zero identity drift.\n"
        "Keep the exact outfit, hairstyle, body proportions, environment and general aesthetic from picture 2.\n"
        "The final image must look like a brand new moment captured in the same place as picture 2, but with a completely different pose and body language.\n"
        "Create a subtle pose variation:\n"
        "the subject must have a slightly different pose, it can be a new arm placement, a new posture and a slightly different camera framing while remaining naturally inside the same environment.\n"
        "The overall body pose do not differs from image 2.\n"
        "Do not recreate or imitate the exact composition of picture 2.\n"
        "The output must feel like another spontaneous photo taken a few moments later in the same location, with potentially a different facial expression.\n"
        "The face must remain fully visible, perfectly recognizable and tack-sharp.\n"
        "Preserve natural facial asymmetry and realistic facial geometry from picture 1.\n"
        "Do not beautify, stylize or alter the identity.\n"
        "Lighting and rendering:\n"
        "The lighting on the face, skin and hair must perfectly match the lighting conditions of picture 2.\n"
        "The face must inherit the same RAW smartphone photo rendering as the rest of the body and scene.\n"
        "No pasted-face effect.\n"
        "No floating face effect.\n"
        "No mismatched skin tones.\n"
        "No incorrect head scale or head placement.\n"
        "Camera/look:\n"
        "Amateur smartphone raw crispness.\n"
        "Shot with a wide-angle lens (24mm equivalent).\n"
        "No portrait mode.\n"
        "Deep focus, entire scene sharp and detailed.\n"
        "Background fully in focus.\n"
        "No bokeh, no shallow depth of field, no motion blur.\n"
        "No haze, no bloom, no glow, no beauty filter, no skin smoothing.\n"
        "Extremely detailed realistic skin texture:\n"
        "visible pores, subtle natural dermal texture, slight natural skin irregularities, realistic sensor grain.\n"
        "Clean skin but not artificially perfect.\n"
        "8K native resolution.\n"
        "Ultra detailed.\n"
        "Photorealistic.\n"
        "Natural RAW photo aesthetic.\n"
        "If anything is uncertain, always prioritize picture 1 for facial identity and picture 2 for environment, outfit and composition logic."
    ),
    "Normal": (
        "Picture 1 defines the identity, facial anatomy, facial proportions, skin texture and hair characteristics of the subject.\n"
        "Picture 2 defines the environment, outfit, camera framing, pose, perspective and overall scene composition.\n"
        "Generate one photorealistic image of the exact same person from picture 1 naturally existing inside the scene of picture 2.\n"
        "The subject must preserve the exact facial likeness from picture 1:\n"
        "same eye shape, eye spacing, eyelids, nose shape, lips, jawline, cheek structure, facial proportions, skin tone and hair characteristics.\n"
        "Maintain extremely high identity consistency with picture 1.\n"
        "Zero identity drift.\n"
        "Keep the exact outfit, hairstyle, body proportions, environment and general aesthetic from picture 2.\n"
        "The final image must look like a brand new moment captured in the same place as picture 2, but with a different pose.\n"
        "Create a pose variation:\n"
        "the subject must have a different pose, it can be a new body positioning, a new arm placement, a new posture and a slightly different camera angle/framing while remaining naturally inside the same environment.\n"
        "Always generate a natural and realistic pose, accorded to instagram vibes.\n"
        "Do not recreate or imitate the exact composition of picture 2.\n"
        "The output must feel like another spontaneous photo taken a few moments later in the same location, with potentially a different facial expression.\n"
        "The face must remain fully visible, perfectly recognizable and tack-sharp.\n"
        "Preserve natural facial asymmetry and realistic facial geometry from picture 1.\n"
        "Do not beautify, stylize or alter the identity.\n"
        "Lighting and rendering:\n"
        "The lighting on the face, skin and hair must perfectly match the lighting conditions of picture 2.\n"
        "The face must inherit the same RAW smartphone photo rendering as the rest of the body and scene.\n"
        "No pasted-face effect.\n"
        "No floating face effect.\n"
        "No mismatched skin tones.\n"
        "No incorrect head scale or head placement.\n"
        "Camera/look:\n"
        "Amateur smartphone raw crispness.\n"
        "Shot with a wide-angle lens (24mm equivalent).\n"
        "No portrait mode.\n"
        "Deep focus, entire scene sharp and detailed.\n"
        "Background fully in focus.\n"
        "No bokeh, no shallow depth of field, no motion blur.\n"
        "No haze, no bloom, no glow, no beauty filter, no skin smoothing.\n"
        "Extremely detailed realistic skin texture:\n"
        "visible pores, subtle natural dermal texture, slight natural skin irregularities, realistic sensor grain.\n"
        "Clean skin but not artificially perfect.\n"
        "8K native resolution.\n"
        "Ultra detailed.\n"
        "Photorealistic.\n"
        "Natural RAW photo aesthetic.\n"
        "If anything is uncertain, always prioritize picture 1 for facial identity and picture 2 for environment, outfit and composition logic."
    ),
    "Heavy": (
        "Picture 1 defines the identity, facial anatomy, facial proportions, skin texture and hair characteristics of the subject.\n"
        "Picture 2 defines the environment, outfit, camera framing, pose, perspective and overall scene composition.\n"
        "Generate one photorealistic image of the exact same person from picture 1 naturally existing inside the scene of picture 2.\n"
        "The subject must preserve the exact facial likeness from picture 1:\n"
        "same eye shape, eye spacing, eyelids, nose shape, lips, jawline, cheek structure, facial proportions, skin tone and hair characteristics.\n"
        "Maintain extremely high identity consistency with picture 1.\n"
        "Zero identity drift.\n"
        "Keep the exact outfit, hairstyle, body proportions, environment and general aesthetic from picture 2.\n"
        "The final image must look like a brand new moment captured in the same place as picture 2, but with a completely different pose and body language.\n"
        "Create a strong pose variation:\n"
        "the subject must have a fully new pose, new body positioning, new arm placement, new posture and different camera angle/framing while remaining naturally inside the same environment.\n"
        "Always generate a natural and realistic pose, accorded to instagram vibes.\n"
        "Do not recreate or imitate the exact composition of picture 2. SHE MUST BE LOCATED SOMEWHERE ELSE IN THE SCENE, with a strong pose and composition variation.\n"
        "The output must feel like another spontaneous photo taken a few moments later in the same location, but somewhere else in this location.\n"
        "The face must remain fully visible, perfectly recognizable and tack-sharp.\n"
        "Preserve natural facial asymmetry and realistic facial geometry from picture 1.\n"
        "Do not beautify, stylize or alter the identity.\n"
        "Lighting and rendering:\n"
        "The lighting on the face, skin and hair must perfectly match the lighting conditions of picture 2.\n"
        "The face must inherit the same RAW smartphone photo rendering as the rest of the body and scene.\n"
        "No pasted-face effect.\n"
        "No floating face effect.\n"
        "No mismatched skin tones.\n"
        "No incorrect head scale or head placement.\n"
        "Camera/look:\n"
        "Amateur smartphone raw crispness.\n"
        "Shot with a wide-angle lens (24mm equivalent).\n"
        "No portrait mode.\n"
        "Deep focus, entire scene sharp and detailed.\n"
        "Background fully in focus.\n"
        "No bokeh, no shallow depth of field, no motion blur.\n"
        "No haze, no bloom, no glow, no beauty filter, no skin smoothing.\n"
        "Extremely detailed realistic skin texture:\n"
        "visible pores, subtle natural dermal texture, slight natural skin irregularities, realistic sensor grain.\n"
        "Clean skin but not artificially perfect.\n"
        "8K native resolution.\n"
        "Ultra detailed.\n"
        "Photorealistic.\n"
        "Natural RAW photo aesthetic.\n"
        "If anything is uncertain, always prioritize picture 1 for facial identity and picture 2 for environment, outfit and composition logic."
    ),
}

from app.schemas.image import ImageGenerationRequest

def generate_fallback_image(prompt: str, preset: str, count_idx: int, image_2: Optional[str] = None) -> Path:
    """
    Creates a high-quality PIL image that composites the face from image_1 (identity reference from
    app/assets/gal.png) onto image_2 (scene/composition reference). If image_2 is not provided,
    it falls back to a dynamically generated gradient landscape.
    """
    from PIL import ImageFont
    
    # Ensure temp dir exists
    temp_dir = Path("temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load image_1 (identity reference face)
    image_1_path = Path("app/assets/gal.png")
    try:
        im1 = Image.open(image_1_path).convert("RGBA")
    except Exception as e:
        logger.warning(f"Failed to load image_1 from {image_1_path}: {e}")
        # Create a fallback face reference
        im1 = Image.new("RGBA", (1024, 1024), (255, 200, 200, 255))
        draw_im1 = ImageDraw.Draw(im1)
        draw_im1.ellipse([300, 300, 700, 700], fill=(255, 255, 255, 255))
        draw_im1.text((350, 450), "Identity [gal.png]", fill=(0, 0, 0))
        
    # 2. Load or generate image_2 (scene reference)
    im2 = None
    if image_2:
        try:
            if image_2.startswith("http"):
                r = httpx.get(image_2, timeout=10.0)
                from io import BytesIO
                im2 = Image.open(BytesIO(r.content)).convert("RGBA")
            else:
                im2 = Image.open(Path(image_2)).convert("RGBA")
            logger.info(f"Loaded image_2 scene successfully.")
        except Exception as e:
            logger.warning(f"Failed to load image_2 '{image_2}', using gradient: {e}")
            im2 = None
            
    if im2 is None:
        # Create a gradient scene background
        im2 = Image.new("RGBA", (1024, 1024))
        draw_im2 = ImageDraw.Draw(im2)
        color_start = (random.randint(20, 60), random.randint(20, 60), random.randint(100, 200), 255)
        color_end = (random.randint(150, 250), random.randint(50, 150), random.randint(50, 150), 255)
        for y in range(1024):
            r = int(color_start[0] + (color_end[0] - color_start[0]) * (y / 1024))
            g = int(color_start[1] + (color_end[1] - color_start[1]) * (y / 1024))
            b = int(color_start[2] + (color_end[2] - color_start[2]) * (y / 1024))
            draw_im2.line([(0, y), (1024, y)], fill=(r, g, b, 255))
            
    # Resize both to standard 1024x1024
    target_size = (1024, 1024)
    im1 = im1.resize(target_size, Image.Resampling.LANCZOS)
    im2 = im2.resize(target_size, Image.Resampling.LANCZOS)
    
    # Create feathered circular mask to extract the face from im1 (gal.png)
    mask = Image.new("L", target_size, 0)
    draw_mask = ImageDraw.Draw(mask)
    cx, cy, r = 512, 350, 200
    draw_mask.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(15))
    
    # Blend/Composite: place face of im1 onto scene of im2
    composite = Image.composite(im1, im2, mask)
    
    # Draw overlays
    draw = ImageDraw.Draw(composite)
    font = None
    try:
        font = ImageFont.truetype("Helvetica", size=30)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", size=30)
        except Exception:
            pass

    text_content = f"Model: Nano Banana Pro\nProvider: VERTEX (PIL Blend)\nPreset: {preset}\nPrompt: {prompt[:50]}...\nVariation: {count_idx + 1}"
    
    draw.text((50, 50), "AI RE-POSED & BLENDED IMAGE", fill=(255, 255, 255), font=font)
    draw.text((50, 100), text_content, fill=(255, 255, 255), font=font)
    draw.text((50, 950), f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}", fill=(200, 200, 200), font=font)
    
    output_path = temp_dir / f"generated_{int(time.time() * 1000)}_{count_idx + 1}.png"
    composite.convert("RGB").save(output_path, "PNG")
    return output_path


async def generate_reposed_image(payload: ImageGenerationRequest) -> dict:
    """
    Generates images using the ComfyUI NanoBananaAIO engine,
    statically loading image_1 from app/assets/gal.png and optionally blending with image_2.
    If NanoBananaAIO execution fails, falls back to the local PIL blender.
    """
    import torch
    import numpy as np
    from app.services.nano_banana_aio import NanoBananaAIO, tensor_to_pil

    name_to_image_map = {
        "gal": Path("app/assets/gal.png"),
    }
    identity = (payload.identity_name or "gal").lower().strip()
    image_1_path = name_to_image_map.get(identity, Path("app/assets/gal.png"))
    count = max(1, min(payload.count, 10))

    # Compile the final prompt exactly like the ComfyUI node
    directive = _PRESET_DIRECTIVES.get(payload.preset, "")
    base = payload.prompt.strip()
    final_prompt = (base + "\n\n" + directive) if base else directive

    if payload.is_selfie:
        final_prompt += (
            "\n\nIt's a selfie, smartphone perspective view, smartphone is not visible, "
            "imperfect framing, slightly angled framing."
        )

    if payload.is_mirror_selfie:
        final_prompt += (
            "\n\nIt's a mirror selfie, the smartphone must be hold next to her face, "
            "the smartphone does not hide her face. We can see the back of the phone, "
            "and we do not see the screen of the phone."
        )

    # Help: load PIL image to PyTorch tensor [1, H, W, 3]
    def pil_to_tensor(pil_img):
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        arr = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    # 1. Load image_1 as tensor (run in thread since Image.open/numpy conversion are CPU-bound)
    try:
        im1 = await asyncio.to_thread(Image.open, image_1_path)
        tensor_image_1 = await asyncio.to_thread(pil_to_tensor, im1)
    except Exception as e:
        logger.error(f"Failed to load image_1 from {image_1_path}: {e}")
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail=f"Identity reference image image_1 not found or corrupted: {e}"
        )

    # 2. Load image_2 as tensor
    tensor_image_2 = None
    if payload.image_2:
        try:
            if payload.image_2.startswith("http"):
                # Use aiohttp to download the image asynchronously
                async with aiohttp.ClientSession() as session:
                    async with session.get(payload.image_2, timeout=10.0) as r:
                        if r.status != 200:
                            raise FileNotFoundError(f"HTTP GET returned status code {r.status}")
                        content = await r.read()
                
                # Decode image in thread pool
                im2 = await asyncio.to_thread(lambda: Image.open(io.BytesIO(content)).convert("RGBA"))
            else:
                im2 = await asyncio.to_thread(lambda: Image.open(Path(payload.image_2)).convert("RGBA"))
                
            tensor_image_2 = await asyncio.to_thread(pil_to_tensor, im2)
        except Exception as e:
            logger.error(f"Failed to load image_2 '{payload.image_2}': {e}")
            from fastapi import HTTPException
            raise HTTPException(
                status_code=404,
                detail=f"Scene/pose reference image image_2 not found or corrupted: {e}"
            )

    generated_images = []
    source = ""

    # Run NanoBananaAIO
    try:
        if tensor_image_1 is None:
            raise ValueError("image_1 (gal.png) is missing or corrupted")

        # Determine aspect ratio
        ar = payload.aspect_ratio
        if ar == "auto":
            ref = tensor_image_2 if tensor_image_2 is not None else tensor_image_1
            _sh = ref.shape
            _h, _w = (_sh[-3], _sh[-2]) if len(_sh) == 4 else (_sh[0], _sh[1])
            _avg = _w / _h
            _AR_SUPPORTED = [
                ("1:1",  1/1),  ("2:3",  2/3),  ("3:2",  3/2),
                ("3:4",  3/4),  ("4:3",  4/3),  ("4:5",  4/5),
                ("5:4",  5/4),  ("9:16", 9/16), ("16:9", 16/9),
                ("21:9", 21/9),
            ]
            ar = min(_AR_SUPPORTED, key=lambda x: abs(x[1] - _avg))[0]
            logger.info(f"Auto Aspect Ratio detected: {ar}")

        aio = NanoBananaAIO()
        logger.info(f"Running NanoBananaAIO.generate_unified...")
        # Await the natively async generator method
        result = await aio.generate_unified(
            provider                 = "VERTEX",
            prompt                   = final_prompt,
            negative_prompt          = "",
            image_size               = payload.image_size,
            gemini_api_key           = "",
            wavespeed_api_key        = "",
            kie_api_key              = "",
            fal_api_key              = "",
            vertex_json_folder       = payload.vertex_json_folder,
            disable_safety_threshold = payload.disable_safety_threshold,
            model                    = "Nano Banana Pro",
            batch_size               = count,
            use_search               = False,
            system_instructions      = None,
            aspect_ratio             = ar,
            temperature              = payload.temperature,
            top_p                    = 0.95,
            fal_safety_tolerance     = "4",
            fal_enable_web_search    = False,
            image_1                  = tensor_image_1,
            image_2                  = tensor_image_2,
            video_mode_enabled       = False,
            face_swap_enabled        = False,
            breast_refiner_enabled   = False,
            low_neck_enabled         = False,
            face_expression          = "Neutral",
            gpt2_image_quality       = "high",
        )

        # Extract tensors from result
        t = result[0]
        if t is not None and list(t.shape) == [1, 64, 64, 3] and float(t.max()) < 0.01:
            raise RuntimeError("NanoBananaAIO returned error dummy image")

        extracted_tensors = []
        if t is not None:
            if t.dim() == 4:
                for b in range(t.shape[0]):
                    extracted_tensors.append(t[b])
            else:
                extracted_tensors.append(t)

        temp_dir = Path("temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        for i, img_tensor in enumerate(extracted_tensors):
            # Run PIL image creation & file saving in a thread pool (CPU & I/O bound)
            def _save_task(tns, idx):
                pil_img = tensor_to_pil(tns)
                out_path = temp_dir / f"generated_{int(time.time() * 1000)}_{idx + 1}.png"
                pil_img.save(out_path, format="PNG")
                return str(out_path)
            
            saved_path_str = await asyncio.to_thread(_save_task, img_tensor, i)
            generated_images.append(saved_path_str)

        if len(generated_images) < count:
            raise RuntimeError(f"Requested {count} images, but only generated {len(generated_images)} via NanoBananaAIO.")

        source = "NanoBananaAIO"
        logger.info(f"Successfully generated {len(generated_images)} images via NanoBananaAIO.")
    except Exception as e:
        logger.warning(f"NanoBananaAIO generation failed, falling back to PIL image blender. Error: {e}")
        
        # ── Fallback PIL Generation (CPU bound) ──
        generated_images = []
        for i in range(count):
            out_path = await asyncio.to_thread(
                generate_fallback_image, payload.prompt, payload.preset, i, payload.image_2
            )
            generated_images.append(str(out_path))
        source = "PIL Fallback Generator"

    # ── Upload to GCS if configured ──
    from app.core.config import settings
    if settings.GCS_BUCKET_NAME:
        import hashlib
        from datetime import datetime, timezone
        from app.services.gcs import upload_file_to_gcs
        
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        prompt_hash = hashlib.sha256(payload.prompt.encode("utf-8")).hexdigest()[:10]
        gcs_prefix = f"generated_images/{date_str}/{prompt_hash}"
        
        gcs_media_files = []
        for file_path_str in generated_images:
            local_file = Path(file_path_str)
            
            # Compress image losslessly (CPU bound)
            compressed_file = await asyncio.to_thread(compress_image_lossless, local_file)
            
            gcs_path = f"{gcs_prefix}/{compressed_file.name}"
            # GCS SDK call is blocking I/O, run in thread pool
            gcs_uri = await asyncio.to_thread(upload_file_to_gcs, compressed_file, gcs_path)
            if gcs_uri:
                gcs_media_files.append(gcs_uri)
                # Cleanup local file (I/O bound)
                try:
                    if compressed_file.exists():
                        await asyncio.to_thread(compressed_file.unlink)
                except Exception as e:
                    logger.warning(f"Failed to delete local generated file {compressed_file}: {e}")
            else:
                raise RuntimeError(f"Failed to upload generated image {compressed_file.name} to GCS.")
        
        generated_images = gcs_media_files

    return {
        "status": "success",
        "generated_images": generated_images,
        "message": f"Successfully generated images via '{source}'. Final prompt used:\n\n{final_prompt}"
    }
