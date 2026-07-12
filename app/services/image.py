import logging
from pathlib import Path
from PIL import Image

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
