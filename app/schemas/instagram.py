from typing import List, Optional
from pydantic import BaseModel


class InstagramDownloadRequest(BaseModel):
    url: str
    max_items: int = 10
    force_refresh: bool = True


class InstagramFileDetail(BaseModel):
    index: int
    name: str
    path: str


class InstagramDownloadSummary(BaseModel):
    mode: str
    downloaded_count: int
    out_dir: str
    files: List[InstagramFileDetail]
    order_source: Optional[str] = None
    required_min_count: Optional[int] = None
    returncode: Optional[int] = None
    elapsed_seconds: Optional[float] = None
    single_instagram_post: Optional[bool] = None
    cookies_used: Optional[bool] = None
    range_applied: Optional[bool] = None
    max_items: Optional[int] = None


class InstagramDownloadResponse(BaseModel):
    status: str = "success"
    original_url: str
    download_url: str
    selected_image_index: int
    url_img_index_override: Optional[int] = None
    download_dir: str
    media_files: List[str]
    summary: InstagramDownloadSummary
