from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse, urlunparse

import requests

from app.core.config import settings
from app.services.gcs import list_gcs_files, upload_file_to_gcs, delete_gcs_files
from app.services.image import compress_image_lossless

# Setup module logger
logger = logging.getLogger("app.services.instagram")

MEDIA_EXTS = {
    # Images
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff",
    # Videos
    ".mp4", ".webm", ".mov", ".m4v", ".avi"
}


def log(message: str) -> None:
    logger.info(f"[SocialLogic] {message}")


def ensure_gallery_dl() -> bool:
    try:
        import gallery_dl  # noqa: F401
        version = getattr(gallery_dl, "__version__", "unknown")
        log(f"gallery-dl ready. version={version}")
        return True
    except ImportError:
        log("gallery-dl not found — installing...")

    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "gallery-dl",
                "--quiet",
            ],
            timeout=120,
        )
        log("gallery-dl installed successfully.")
        return True
    except Exception as exc:
        log(f"gallery-dl install failed: {exc}")
        return False


def is_direct_image_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    ext = os.path.splitext(path)[1]
    return ext in MEDIA_EXTS


def is_instagram_single_post(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower().strip("/")

    if "instagram.com" not in host:
        return False

    return path.startswith("p/") or path.startswith("reel/") or path.startswith("tv/")


def strip_query_and_fragment(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse(parsed._replace(query="", fragment=""))


def extract_instagram_img_index(url: str) -> Optional[int]:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    if "img_index" not in query:
        return None

    try:
        value = int(query["img_index"][0])
        if value <= 0:
            return None
        return value - 1
    except Exception:
        return None


def choose_image_index_from_url(source_url: str) -> tuple[int, Optional[int]]:
    url_img_index = extract_instagram_img_index(source_url)

    if url_img_index is not None:
        log(f"URL img_index detected. selected_image_index={url_img_index}")
        return url_img_index, url_img_index

    log("No img_index in URL. Defaulting to first image.")
    return 0, None


def safe_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    raw = parsed.netloc + parsed.path
    raw = raw.strip("/").replace("/", "_").replace(":", "_")

    if not raw:
        raw = "download"

    url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{raw[:80]}_{url_hash}"


def pick_extension_from_response(response: requests.Response, url: str) -> str:
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()

    content_type_to_ext = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
        "image/gif": ".gif",
        "image/tiff": ".tiff",
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "video/quicktime": ".mov",
    }

    if content_type in content_type_to_ext:
        return content_type_to_ext[content_type]

    path = urlparse(url).path
    ext = os.path.splitext(unquote(path))[1].lower()

    if ext in MEDIA_EXTS:
        return ext

    return ".png"


def describe_files_short(files: list[str]) -> list[dict]:
    return [
        {
            "index": index,
            "name": Path(file_path).name,
            "path": os.path.abspath(file_path),
        }
        for index, file_path in enumerate(files)
    ]


def collect_media_files(root_folder: str | Path) -> list[str]:
    root_path = Path(root_folder)

    if not root_path.exists():
        return []

    files: list[str] = []

    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in MEDIA_EXTS:
            files.append(str(path))

    return sorted(files)


def parse_gallery_dl_stdout_media_order(
    stdout: str,
    out_dir: str | Path,
) -> list[str]:
    out_dir = str(Path(out_dir))
    ordered: list[str] = []
    seen: set[str] = set()

    for raw_line in stdout.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if not line.startswith(out_dir):
            continue

        path = Path(line)

        if path.suffix.lower() not in MEDIA_EXTS:
            continue

        path_str = str(path)

        if path_str in seen:
            continue

        if path.exists():
            ordered.append(path_str)
            seen.add(path_str)

    return ordered


def merge_ordered_with_remaining(
    ordered_files: list[str],
    all_files: list[str],
) -> list[str]:
    seen = set(ordered_files)
    merged = list(ordered_files)

    for file_path in all_files:
        if file_path not in seen:
            merged.append(file_path)
            seen.add(file_path)

    return merged


def wait_for_media_stable(
    folder: str | Path,
    min_count: int = 1,
    timeout_seconds: int = 30,
    stable_checks_required: int = 3,
    sleep_seconds: float = 0.5,
) -> list[str]:
    deadline = time.time() + timeout_seconds
    last_files: list[str] = []
    stable_checks = 0

    while time.time() < deadline:
        files = collect_media_files(folder)

        same_as_last = files == last_files
        enough = len(files) >= min_count

        if enough and same_as_last:
            stable_checks += 1
        else:
            stable_checks = 0

        if stable_checks >= stable_checks_required:
            return files

        last_files = files
        time.sleep(sleep_seconds)

    return collect_media_files(folder)


def get_base_directory(save_to: str) -> str:
    # Use relative workspace downloads folder
    base_dir = Path("downloads")
    
    if save_to == "input":
        target = base_dir / "input"
    elif save_to == "output":
        target = base_dir / "output"
    else:
        target = base_dir / "temp"

    target.mkdir(parents=True, exist_ok=True)
    return str(target.resolve())


def clear_directory(path: str | Path) -> None:
    path = Path(path)
    log(f"Clearing download folder: {path}")
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def download_direct_image(
    url: str,
    out_dir: str | Path,
    timeout: int,
) -> tuple[list[str], str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log(f"Downloading direct image URL.")
    log(f"Download folder: {out_path}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=timeout,
        stream=True,
        allow_redirects=True,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()

    if not content_type.startswith("image/") and not content_type.startswith("video/"):
        raise ValueError(f"URL did not return media. Content-Type was: {content_type}")

    ext = pick_extension_from_response(response, url)
    filename = f"direct_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]}{ext}"
    local_path = out_path / filename

    with local_path.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)

    media_files = wait_for_media_stable(
        folder=out_path,
        min_count=1,
        timeout_seconds=10,
        stable_checks_required=2,
        sleep_seconds=0.25,
    )

    summary = {
        "mode": "direct-image",
        "downloaded_count": len(media_files),
        "out_dir": str(out_path.resolve()),
        "files": describe_files_short(media_files),
    }

    return media_files, json.dumps(summary, indent=2, ensure_ascii=False)


def build_gallery_dl_command(
    url: str,
    out_dir: str | Path,
    cookies_file: str,
    max_items: int,
) -> tuple[list[str], dict]:
    out_dir = str(out_dir)
    single_instagram_post = is_instagram_single_post(url)

    cmd = [
        sys.executable,
        "-m",
        "gallery_dl",
        "--dest",
        out_dir,
        "--no-mtime",
    ]

    cookies_file = (cookies_file or "").strip()
    cookies_used = False

    if cookies_file:
        if not os.path.isfile(cookies_file):
            raise ValueError(f"cookies_file does not exist: {cookies_file}")

        cmd += ["--cookies", cookies_file]
        cookies_used = True

    range_applied = False

    if max_items > 0 and not single_instagram_post:
        cmd += ["--range", f"1-{max_items}"]
        range_applied = True

    cmd.append(url)

    metadata = {
        "single_instagram_post": single_instagram_post,
        "cookies_used": cookies_used,
        "range_applied": range_applied,
        "max_items": max_items,
        "out_dir": out_dir,
    }

    return cmd, metadata


def download_with_gallery_dl(
    url: str,
    out_dir: str | Path,
    cookies_file: str,
    max_items: int,
    timeout: int,
    selected_image_index: int,
) -> tuple[list[str], str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not ensure_gallery_dl():
        raise RuntimeError("gallery-dl could not be verified or installed.")

    cmd, metadata = build_gallery_dl_command(
        url=url,
        out_dir=out_path,
        cookies_file=cookies_file,
        max_items=max_items,
    )

    if metadata["cookies_used"]:
        log("Using cookies file: provided")
    else:
        log("Using cookies file: none")

    if metadata["single_instagram_post"]:
        log("Single Instagram post detected. Not applying --range.")
    elif metadata["range_applied"]:
        log(f"Applying --range 1-{max_items}")
    else:
        log("No --range applied.")

    log(f"Running gallery-dl into: {out_path}")

    start = time.time()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=max(timeout, 120),
    )

    elapsed = time.time() - start

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    stdout_tail = "\n".join(stdout.strip().splitlines()[-20:]) if stdout else ""
    stderr_tail = "\n".join(stderr.strip().splitlines()[-20:]) if stderr else ""

    log(f"gallery-dl finished. returncode={result.returncode}, elapsed={elapsed:.2f}s")

    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"gallery-dl failed with code {result.returncode}\n\n"
            f"STDOUT TAIL:\n{stdout_tail}\n\n"
            f"STDERR TAIL:\n{stderr_tail}"
        )

    min_needed = max(1, selected_image_index + 1)

    stable_files = wait_for_media_stable(
        folder=out_path,
        min_count=min_needed,
        timeout_seconds=45,
        stable_checks_required=4,
        sleep_seconds=0.5,
    )

    if not stable_files:
        raise RuntimeError(
            "gallery-dl finished but no media files were found.\n\n"
            f"STDOUT TAIL:\n{stdout_tail}\n\n"
            f"STDERR TAIL:\n{stderr_tail}"
        )

    stdout_ordered_files = parse_gallery_dl_stdout_media_order(
        stdout=stdout,
        out_dir=out_path,
    )

    if stdout_ordered_files:
        media_files = merge_ordered_with_remaining(
            ordered_files=stdout_ordered_files,
            all_files=stable_files,
        )
        order_source = "gallery-dl stdout order"
    else:
        media_files = stable_files
        order_source = "filesystem sorted fallback"
        log("Warning: could not parse gallery-dl stdout order; using filesystem order.")

    log(f"Media files found: {len(media_files)}")
    log(f"Order source: {order_source}")

    summary = {
        "mode": "gallery-dl",
        "order_source": order_source,
        "downloaded_count": len(media_files),
        "required_min_count": min_needed,
        "out_dir": str(out_path.resolve()),
        "returncode": result.returncode,
        "elapsed_seconds": round(elapsed, 2),
        "files": describe_files_short(media_files),
        **metadata,
    }

    return media_files, json.dumps(summary, indent=2, ensure_ascii=False)


def resolve_download(
    source_url: str,
    save_to: str,
    cookies_file: str,
    max_items: int,
    timeout: int,
    force_refresh: bool,
    selected_image_index: int,
) -> tuple[str, str, list[str], str, str]:
    original_source_url = (source_url or "").strip()

    if not original_source_url:
        raise ValueError("source_url is empty.")

    download_source_url = strip_query_and_fragment(original_source_url)

    log(f"Original URL: {original_source_url}")
    log(f"Download URL: {download_source_url}")
    log(f"Selected image index: {selected_image_index}")
    log(f"Save target: {save_to}")

    folder_name = safe_name_from_url(download_source_url)
    min_needed = max(1, selected_image_index + 1)

    # 1. If Google Cloud Storage is enabled
    if settings.GCS_BUCKET_NAME:
        gcs_prefix = f"social_downloads/{folder_name}"
        
        # Check cache if not forcing refresh
        if not force_refresh:
            cached_files = list_gcs_files(prefix=gcs_prefix)
            if cached_files and len(cached_files) >= min_needed:
                log(f"Using GCS cached files. count={len(cached_files)}")
                summary = {
                    "mode": "gcs-cache",
                    "order_source": "gcs prefix search",
                    "downloaded_count": len(cached_files),
                    "required_min_count": min_needed,
                    "out_dir": f"gs://{settings.GCS_BUCKET_NAME}/{gcs_prefix}",
                    "files": [
                        {
                            "index": idx,
                            "name": Path(f).name,
                            "path": f,
                        }
                        for idx, f in enumerate(cached_files)
                    ],
                }
                return (
                    original_source_url,
                    download_source_url,
                    cached_files,
                    json.dumps(summary, indent=2, ensure_ascii=False),
                    f"gs://{settings.GCS_BUCKET_NAME}/{gcs_prefix}",
                )

        # Force refresh or GCS cache miss -> Delete existing if force_refresh is True
        if force_refresh:
            delete_gcs_files(prefix=gcs_prefix)

        # Download to a temporary folder in the pod
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            if is_direct_image_url(download_source_url):
                media_files, summary_str = download_direct_image(
                    url=download_source_url,
                    out_dir=temp_path,
                    timeout=timeout,
                )
            else:
                media_files, summary_str = download_with_gallery_dl(
                    url=download_source_url,
                    out_dir=temp_path,
                    cookies_file=cookies_file,
                    max_items=max_items,
                    timeout=timeout,
                    selected_image_index=selected_image_index,
                )

            # Compress downloaded images losslessly
            compressed_media_files = []
            for file_path_str in media_files:
                local_file = Path(file_path_str)
                compressed_file = compress_image_lossless(local_file)
                compressed_media_files.append(compressed_file)

            # Upload compressed files to GCS
            gcs_media_files = []
            for local_file in compressed_media_files:
                gcs_path = f"{gcs_prefix}/{local_file.name}"
                gcs_uri = upload_file_to_gcs(local_file, gcs_path)
                if gcs_uri:
                    gcs_media_files.append(gcs_uri)
                else:
                    raise RuntimeError(f"Failed to upload media file {local_file.name} to GCS.")

            original_summary = json.loads(summary_str)
            summary = {
                **original_summary,
                "mode": f"gcs-{original_summary.get('mode', 'download')}",
                "out_dir": f"gs://{settings.GCS_BUCKET_NAME}/{gcs_prefix}",
                "files": [
                    {
                        "index": idx,
                        "name": Path(f).name,
                        "path": f,
                    }
                    for idx, f in enumerate(gcs_media_files)
                ],
            }
            return (
                original_source_url,
                download_source_url,
                gcs_media_files,
                json.dumps(summary, indent=2, ensure_ascii=False),
                f"gs://{settings.GCS_BUCKET_NAME}/{gcs_prefix}",
            )

    # 2. Local fallback storage
    base_dir = get_base_directory(save_to)
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    download_dir = Path(base_dir) / "social_downloads" / folder_name
    download_dir.mkdir(parents=True, exist_ok=True)

    log(f"Download folder: {download_dir}")

    if force_refresh:
        clear_directory(download_dir)

    cached_files = wait_for_media_stable(
        folder=download_dir,
        min_count=min_needed,
        timeout_seconds=2,
        stable_checks_required=1,
        sleep_seconds=0.25,
    )

    if cached_files and not force_refresh and len(cached_files) >= min_needed:
        log(f"Using cached files. count={len(cached_files)}")
        log("Note: cached files use filesystem order. Use force_refresh=True if carousel order looks wrong.")

        summary = {
            "mode": "cache",
            "order_source": "filesystem sorted cache order",
            "downloaded_count": len(cached_files),
            "required_min_count": min_needed,
            "out_dir": str(download_dir.resolve()),
            "files": describe_files_short(cached_files),
        }

        return (
            original_source_url,
            download_source_url,
            cached_files,
            json.dumps(summary, indent=2, ensure_ascii=False),
            str(download_dir.resolve()),
        )

    if is_direct_image_url(download_source_url):
        media_files, summary_str = download_direct_image(
            url=download_source_url,
            out_dir=download_dir,
            timeout=timeout,
        )
    else:
        media_files, summary_str = download_with_gallery_dl(
            url=download_source_url,
            out_dir=download_dir,
            cookies_file=cookies_file,
            max_items=max_items,
            timeout=timeout,
            selected_image_index=selected_image_index,
        )

    # Compress downloaded images losslessly for local fallback too
    compressed_media_files = []
    for file_path_str in media_files:
        local_file = Path(file_path_str)
        compressed_file = compress_image_lossless(local_file)
        compressed_media_files.append(str(compressed_file.resolve()))
    media_files = compressed_media_files

    original_summary = json.loads(summary_str)
    summary = {
        **original_summary,
        "files": describe_files_short(media_files),
    }
    summary_str = json.dumps(summary, indent=2, ensure_ascii=False)

    return original_source_url, download_source_url, media_files, summary_str, str(download_dir.resolve())


def download_instagram_post(
    source_url: str,
    max_items: int = 10,
    force_refresh: bool = True,
) -> dict:
    """
    Downloads all items in an Instagram post or carousel.
    """
    selected_image_index, url_img_index = choose_image_index_from_url(source_url)
    timeout = settings.INSTAGRAM_DOWNLOAD_TIMEOUT
    save_to = settings.INSTAGRAM_SAVE_TO

    cookies_file = ""
    temp_cookies_path = None

    if settings.INSTAGRAM_COOKIES:
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tf:
                tf.write(settings.INSTAGRAM_COOKIES)
                temp_cookies_path = tf.name
            cookies_file = temp_cookies_path
            log(f"Wrote temporary cookies file to {temp_cookies_path}")
        except Exception as e:
            log(f"Failed to create temporary cookies file: {e}")

    try:
        (
            original_source_url,
            download_source_url,
            media_files,
            download_summary,
            download_dir,
        ) = resolve_download(
            source_url=source_url,
            save_to=save_to,
            cookies_file=cookies_file,
            max_items=max_items,
            timeout=timeout,
            force_refresh=force_refresh,
            selected_image_index=selected_image_index,
        )
        parsed_summary = json.loads(download_summary)
        return {
            "original_url": original_source_url,
            "download_url": download_source_url,
            "selected_image_index": selected_image_index,
            "url_img_index_override": url_img_index,
            "download_dir": download_dir,
            "media_files": [f if f.startswith("gs://") else os.path.abspath(f) for f in media_files],
            "summary": parsed_summary,
        }
    finally:
        if temp_cookies_path and os.path.exists(temp_cookies_path):
            try:
                os.unlink(temp_cookies_path)
                log(f"Cleaned up temporary cookies file {temp_cookies_path}")
            except Exception as e:
                log(f"Failed to delete temporary cookies file {temp_cookies_path}: {e}")
