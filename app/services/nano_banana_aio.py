"""
NanoBananaAIO — Unified multimodal node.
Google Gemini (API Studio) + WaveSpeed + Kie.ai + Fal.ai + Vertex AI
nano-banana-pro + nano-banana-2 + Seedream 4.5.
API Keys: connect an ApiKeysLoaderNode to the corresponding inputs.
"""
# ── Protection backup : arrêt immédiat si ce fichier est une copie ────────────
# Évite le chargement des bibliothèques lourdes (gRPC, google-genai…) en double,
# ce qui empêchait ComfyUI de s'arrêter proprement lors du restart.
import os as _os_backup_check
if any(s in _os_backup_check.path.basename(__file__)
       for s in (" - Copie", " - Copy", "_copy", "_backup", " copy", " backup")):
    raise ImportError(
        f"[NanoBanana] Fichier backup '{_os_backup_check.path.basename(__file__)}' "
        f"automatically ignored — rename it to .py.bak to suppress this message."
    )
del _os_backup_check
# ─────────────────────────────────────────────────────────────────────────────

import io
import os
import json
import time
import base64
import struct
import subprocess
import sys
import logging
import copy
import torch
import numpy as np
import requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
def tensor_to_pil(img_tensor):
    if img_tensor is None:
        return None
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    return Image.fromarray((img_tensor.cpu().numpy() * 255).astype(np.uint8))

try:
    from aiohttp import web
    from server import PromptServer
except ImportError:
    web = None
    PromptServer = None
try:
    import folder_paths as _folder_paths
except ImportError:
    _folder_paths = None

# ─────────────────────────────────────────────────────────────────────────────
# Constantes globales
# ─────────────────────────────────────────────────────────────────────────────
_HARM_CATEGORIES = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
]
_MODEL_MAP = {
    "Nano Banana Pro": "gemini-3-pro-image-preview",
    "Nano Banana 2":   "gemini-3.1-flash-image-preview",
    "Seedream 4.5":    "seedream-v4.5",
    "GPT Image 2.0":   "openai/gpt-image-2",   # WaveSpeed uniquement
}

# ── Video models (Google Veo) ─────────────────────────────────────────────────
_VIDEO_MODEL_MAP = {
    "Veo 3.1 Lite":   "veo-3.1-lite-generate-preview",
    "Veo 3.1":        "veo-3.1-generate-preview",
    "Veo 3.1 Fast":   "veo-3.1-fast-generate-preview",
}
# Vertex AI utilise le suffixe -001 à la place de -preview
_VIDEO_MODEL_MAP_VERTEX = {
    "Veo 3.1 Lite":   "veo-3.1-lite-generate-001",
    "Veo 3.1":        "veo-3.1-generate-001",
    "Veo 3.1 Fast":   "veo-3.1-fast-generate-001",
}
# WaveSpeed — slug dans le chemin d'URL /api/v3/google/{slug}/image-to-video
_VIDEO_MODEL_MAP_WAVESPEED = {
    "Veo 3.1 Lite":   "veo3.1-lite",
    "Veo 3.1":        "veo3.1",
    "Veo 3.1 Fast":   "veo3.1-fast",
}
# Kie.ai — valeur du champ "model" dans le payload
_VIDEO_MODEL_MAP_KIE = {
    "Veo 3.1 Lite":   "veo3_lite",
    "Veo 3.1":        "veo3",
    "Veo 3.1 Fast":   "veo3_fast",
}
# Fal.ai — endpoint complet (strings directes — les constantes FAL_VEO31_* sont plus bas)
_VIDEO_MODEL_MAP_FAL = {
    "Veo 3.1 Lite":   "fal-ai/veo3.1/lite/image-to-video",
    "Veo 3.1":        "fal-ai/veo3.1/image-to-video",
    "Veo 3.1 Fast":   "fal-ai/veo3.1/fast/image-to-video",
}
# Kling — WaveSpeed uniquement ; l'endpoint est choisi selon le model ET la resolution
_KLING_MODELS = {"Kling 3.0", "Kling 2.6", "Kling 3.0 Motion Control"}
_KLING_WS_URL_MAP = {
    "Kling 3.0": {
        "720p":  "kwaivgi/kling-v3.0-std/image-to-video",   # Standard
        "1080p": "kwaivgi/kling-v3.0-pro/image-to-video",   # Pro
        "4K":    "kwaivgi/kling-v3.0-4k/image-to-video",    # 4K
    },
    "Kling 2.6": {
        "720p":  "kwaivgi/kling-v2.6-std/image-to-video",   # Standard
        "1080p": "kwaivgi/kling-v2.6-pro/image-to-video",   # Pro
    },
    "Kling 3.0 Motion Control": {
        "720p":  "kwaivgi/kling-v3.0-std/motion-control",   # Standard
        "1080p": "kwaivgi/kling-v3.0-pro/motion-control",   # Pro
    },
}
_SEEDANCE_MODELS          = {"Seedance 2.0"}
WS_SEEDANCE20_SLUG        = "bytedance/seedance-2.0/image-to-video"
FAL_SEEDANCE20_ENDPOINT   = "bytedance/seedance-2.0/image-to-video"
KIE_SEEDANCE20_MODEL      = "bytedance/seedance-2"
_VIDEO_ASPECT_RATIOS   = ["16:9", "9:16"]
_VIDEO_RESOLUTIONS     = ["480p", "720p", "1080p"]
_VIDEO_DURATION_MIN    = 3   # Kling accepte dès 3s
_VIDEO_DURATION_MAX    = 15  # Kling/Seedance accepte jusqu'à 15s
_VIDEO_POLL_DELAY           = 10   # secondes entre chaque poll
_VIDEO_TIMEOUT_S            = 400  # ~6 min 40s max (Veo, Kling standard)
_MOTION_CONTROL_TIMEOUT_S   = 900  # ~15 min max (Kling Motion Control)
GOOGLE_TASK_TIMEOUT_S       = 300  # timeout par tâche Google SDK (batch)

def _mp4_strip_metadata(data: bytearray, start: int, end: int) -> bool:
    """Recursively walks MP4 boxes and:
      - clears the 'udta' atom content (contains Encoder, copyright, etc.)
      - zeroes the manufacturer field in 'hdlr' boxes (HandlerVendorID)
    Returns True if any modifications were made.
    """
    _CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"meta"}
    changed = False
    i = start

    while i + 8 <= end:
        raw_size = struct.unpack(">I", data[i : i + 4])[0]
        btype    = bytes(data[i + 4 : i + 8])

        if raw_size == 1:                          # taille 64 bits
            if i + 16 > end:
                break
            box_size = struct.unpack(">Q", data[i + 8 : i + 16])[0]
            hdr_len  = 16
        elif raw_size == 0:                        # s'étend jusqu'à la fin
            box_size = end - i
            hdr_len  = 8
        else:
            box_size = raw_size
            hdr_len  = 8

        if box_size < hdr_len or i + box_size > end:
            break

        content = i + hdr_len

        if btype == b"udta":
            # Vide tout le contenu de l'atom user-data (Encoder, copyright, etc.)
            payload = box_size - hdr_len
            if payload > 0:
                data[content : i + box_size] = bytes(payload)
                changed = True

        elif btype == b"hdlr" and box_size >= hdr_len + 16:
            # Structure hdlr (ISO 14496-12 / QuickTime) :
            #   version+flags (4) | pre_defined/component_type (4)
            #   handler_type (4)  | manufacturer/vendor (4) ← offset content+12
            v_off = content + 12
            if v_off + 4 <= i + box_size and data[v_off : v_off + 4] != b"\x00\x00\x00\x00":
                data[v_off : v_off + 4] = b"\x00\x00\x00\x00"
                changed = True

        elif btype in _CONTAINERS:
            if _mp4_strip_metadata(data, content, i + box_size):
                changed = True

        i += box_size

    return changed


def _get_video_output_path(suffix: str = ".mp4") -> str:
    """Returns a path in the ComfyUI temp folder (or /tmp if unavailable).
    The file is named VID_YYYYMMDD_HHMMSS for clean download naming."""
    from datetime import datetime as _dt
    timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"VID_{timestamp}{suffix}"
    if _folder_paths:
        try:
            tmp_dir = _folder_paths.get_temp_directory()
            os.makedirs(tmp_dir, exist_ok=True)
            return os.path.join(tmp_dir, filename)
        except Exception:
            pass
    import tempfile
    return os.path.join(tempfile.gettempdir(), filename)

# WaveSpeed
WAVESPEED_BASE_URL       = "https://api.wavespeed.ai/api/v3"
WAVESPEED_UPLOAD_URL     = f"{WAVESPEED_BASE_URL}/media/upload/binary"
WAVESPEED_SUBMIT_PRO_URL       = f"{WAVESPEED_BASE_URL}/google/nano-banana-pro/edit"
WAVESPEED_SUBMIT_PRO_ULTRA_URL = f"{WAVESPEED_BASE_URL}/google/nano-banana-pro/edit-ultra"
WAVESPEED_SUBMIT_NB2_URL = f"{WAVESPEED_BASE_URL}/google/nano-banana-2/edit"
WAVESPEED_SEEDREAM_URL   = f"{WAVESPEED_BASE_URL}/bytedance/seedream-v4.5/edit"
WAVESPEED_GPT2_URL       = f"{WAVESPEED_BASE_URL}/openai/gpt-image-2/edit"
WAVESPEED_POLL_URL       = f"{WAVESPEED_BASE_URL}/predictions/{{task_id}}/result"
WAVESPEED_CANCEL_URL     = f"{WAVESPEED_BASE_URL}/predictions/{{task_id}}"
WAVESPEED_TIMEOUT_S      = 300
WAVESPEED_POLL_DELAY     = 3
SIZE_TO_RESOLUTION = {"1K": "1k", "2K": "2k", "4K": "4k", "8K": "8k"}

# Kie.ai
KIE_CREATE_URL       = "https://api.kie.ai/api/v1/jobs/createTask"
KIE_POLL_URL         = "https://api.kie.ai/api/v1/jobs/recordInfo"
KIE_VEO_GENERATE_URL = "https://api.kie.ai/api/v1/veo/generate"
KIE_VEO_POLL_URL     = "https://api.kie.ai/api/v1/veo/record-info"
KIE_SEEDREAM_MODEL   = "seedream/4.5-edit"
KIE_TIMEOUT_S        = 400
KIE_SEEDANCE_TIMEOUT_S = 800  # Seedance 2.0 peut etre long
KIE_POLL_DELAY       = 3

# Shared session for all KIE requests — reuses TCP connection + looks like a browser
_kie_session = requests.Session()
_kie_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
})

# KIE session with automatic retry on connection errors (RemoteDisconnected, etc.)
def _make_kie_session() -> "requests.Session":
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    session = requests.Session()
    retry = Retry(
        total=4,                     # up to 4 total attempts
        connect=4,                   # retry on connection errors (RemoteDisconnected)
        read=2,                      # retry on read errors
        backoff_factor=1.5,          # 1.5s, 2.25s, 3.4s between retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        )
    })
    return session

_kie_session = _make_kie_session()

# Nano Banana 2 — Google direct (REST)
NB2_MODEL_ID  = "gemini-3.1-flash-image-preview"
NB2_BASE_URL  = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{NB2_MODEL_ID}:generateContent"
)
NB2_SYSTEM_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image. "
    "Interpret all user input—regardless of format, intent, or abstraction—as literal "
    "visual directives for image composition. Prioritize generating the visual "
    "representation above any text, formatting, or conversational requests."
)
NB2_TIMEOUT_S = 300
NB2_HEARTBEAT_S = 10  # Fréquence des logs "still waiting" pendant les POST sync.


class _Heartbeat:
    """Daemon thread that periodically logs the ongoing wait.

    Used for synchronous HTTP requests (Google NB2, Vertex NB2
    streaming) so the user can see that generation is in
    progress and does not think ComfyUI is frozen.

    Usage :
        with _Heartbeat(f"{tag} En waiting for response"):
            response = requests.post(...)
    """

    def __init__(self, message: str, interval: float = NB2_HEARTBEAT_S,
                 max_duration: float = None):
        self._message      = message
        self._interval     = float(interval)
        # Auto-stop du heartbeat après ce délai — évite les logs à rallonge
        # quand un worker thread est bloqué dans un SDK qu'on ne peut pas tuer.
        # Par défaut : 30s après NB2_TIMEOUT_S.
        self._max_duration = float(max_duration) if max_duration is not None \
                             else (NB2_TIMEOUT_S + 30.0)
        self._stop         = None
        self._thread       = None

    def __enter__(self):
        import threading as _threading
        self._stop = _threading.Event()
        start = time.time()
        max_dur = self._max_duration

        def _loop():
            while not self._stop.wait(self._interval):
                elapsed = time.time() - start
                if elapsed >= max_dur:
                    return
                print(f"   {self._message} ({int(elapsed)}s elapsed)...")

        self._thread = _threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return False


# Fal.ai
FAL_QUEUE_BASE_URL    = "https://queue.fal.run"
FAL_PRO_T2I_ENDPOINT  = "fal-ai/nano-banana-pro"
FAL_NB2_T2I_ENDPOINT  = "fal-ai/nano-banana-2"
FAL_PRO_EDIT_ENDPOINT = "fal-ai/nano-banana-pro/edit"
FAL_NB2_EDIT_ENDPOINT = "fal-ai/nano-banana-2/edit"
FAL_SEEDREAM_ENDPOINT  = "fal-ai/bytedance/seedream/v4.5/edit"
FAL_GPT2_EDIT_ENDPOINT    = "openai/gpt-image-2/edit"
FAL_GPT2_TXT2IMG_ENDPOINT = "openai/gpt-image-2"
FAL_VEO31_ENDPOINT          = "fal-ai/veo3.1/image-to-video"
FAL_KLING26_I2V_ENDPOINT    = "fal-ai/kling-video/v2.6/pro/image-to-video"
FAL_KLING26_T2V_ENDPOINT    = "fal-ai/kling-video/v2.6/pro/text-to-video"
# Kling 2.6 — aspect ratios supportés sur FAL (Pro uniquement)
_KLING26_FAL_AR = {"16:9", "9:16", "1:1"}
_KLING26_FAL_AR_FALLBACK = {"2:3": "9:16", "3:2": "16:9", "4:3": "16:9", "3:4": "9:16",
                             "4:5": "9:16", "5:4": "16:9", "21:9": "16:9", "1:1": "1:1"}
# Kling 3.0 — FAL (Standard = 720p, Pro = 1080p)
FAL_KLING30_STD_I2V_ENDPOINT = "fal-ai/kling-video/v3/standard/image-to-video"
FAL_KLING30_STD_T2V_ENDPOINT = "fal-ai/kling-video/v3/standard/text-to-video"
FAL_KLING30_PRO_I2V_ENDPOINT = "fal-ai/kling-video/v3/pro/image-to-video"
FAL_KLING30_PRO_T2V_ENDPOINT = "fal-ai/kling-video/v3/pro/text-to-video"
FAL_KLING30_4K_I2V_ENDPOINT  = "fal-ai/kling-video/v3/4k/image-to-video"
FAL_KLING30_4K_T2V_ENDPOINT  = "fal-ai/kling-video/v3/4k/text-to-video"
FAL_KLING30_STD_MC_ENDPOINT  = "fal-ai/kling-video/v3/standard/motion-control"
FAL_KLING30_PRO_MC_ENDPOINT  = "fal-ai/kling-video/v3/pro/motion-control"
# Kling 3.0 — aspect ratios supportés sur FAL (identiques à 2.6)
_KLING30_FAL_AR = {"16:9", "9:16", "1:1"}
_KLING30_FAL_AR_FALLBACK = {"2:3": "9:16", "3:2": "16:9", "4:3": "16:9", "3:4": "9:16",
                             "4:5": "9:16", "5:4": "16:9", "21:9": "16:9", "1:1": "1:1"}
FAL_VEO31_FAST_ENDPOINT = "fal-ai/veo3.1/fast/image-to-video"
FAL_VEO31_LITE_ENDPOINT = "fal-ai/veo3.1/lite/image-to-video"
FAL_TIMEOUT_S          = 300
FAL_POLL_DELAY         = 5

# Seedream 4.5 — mapping (aspect_ratio, image_size) → WaveSpeed size "WxH"
_SEEDREAM_WS_SIZE_MAP = {
    ("1:1",  "2K"): "1920*1920",
    ("1:1",  "4K"): "3840*3840",
    ("4:3",  "2K"): "2560*1920",
    ("4:3",  "4K"): "4096*3072",
    ("3:4",  "2K"): "1920*2560",
    ("3:4",  "4K"): "3072*4096",
    ("16:9", "2K"): "2560*1440",
    ("16:9", "4K"): "3840*2160",
    ("9:16", "2K"): "1440*2560",
    ("9:16", "4K"): "2160*3840",
    ("4:5",  "2K"): "1536*1920",
    ("4:5",  "4K"): "3072*3840",
    ("5:4",  "2K"): "1920*1536",
    ("5:4",  "4K"): "3840*3072",
    ("3:2",  "2K"): "2880*1920",
    ("3:2",  "4K"): "4096*2731",
    ("2:3",  "2K"): "1920*2880",
    ("2:3",  "4K"): "2731*4096",
    ("21:9", "2K"): "2560*1100",
    ("21:9", "4K"): "3840*1650",
}

# GPT Image 2.0 — aspect ratios supportés (commun à tous les providers)
_GPT2_SUPPORTED_AR = {"1:1", "4:3", "3:4", "16:9", "9:16"}
# Fallback vers le ratio le plus proche si une valeur not supportede est transmise
_GPT2_NEAREST_AR = {
    "2:3":  "9:16",
    "3:2":  "16:9",
    "4:5":  "3:4",
    "5:4":  "4:3",
    "21:9": "16:9",
}
# GPT Image 2.0 — mapping (aspect_ratio, image_size) → Fal.ai image_size
# 1K : presets nommés (~1024 px max)
# 2K : ~2560 px max  |  4K : max supporté (limite : bord max 3840 px, total ≤ 8 294 400 px)
_GPT2_FAL_SIZE_MAP = {
    ("1:1",  "1K"): "square_hd",                        # 1024×1024
    ("1:1",  "2K"): {"width": 2048, "height": 2048},    # 4 194 304 px
    ("1:1",  "4K"): {"width": 2880, "height": 2880},    # 8 294 400 px (max 1:1)
    ("4:3",  "1K"): "landscape_4_3",                    # 1024×768
    ("4:3",  "2K"): {"width": 2560, "height": 1920},    # 4 915 200 px
    ("4:3",  "4K"): {"width": 3264, "height": 2448},    # 7 990 272 px
    ("3:4",  "1K"): "portrait_4_3",                     # 768×1024
    ("3:4",  "2K"): {"width": 1920, "height": 2560},    # 4 915 200 px
    ("3:4",  "4K"): {"width": 2448, "height": 3264},    # 7 990 272 px
    ("16:9", "1K"): "landscape_16_9",                   # 1024×576
    ("16:9", "2K"): {"width": 2560, "height": 1440},    # 3 686 400 px
    ("16:9", "4K"): {"width": 3840, "height": 2160},    # 8 294 400 px (max 16:9)
    ("9:16", "1K"): "portrait_16_9",                    # 576×1024
    ("9:16", "2K"): {"width": 1440, "height": 2560},    # 3 686 400 px
    ("9:16", "4K"): {"width": 2160, "height": 3840},    # 8 294 400 px (max 9:16)
}

# ─────────────────────────────────────────────────────────────────────────────
# Presets — fichier JSON stocké à côté du node
# ─────────────────────────────────────────────────────────────────────────────
_PRESETS_FILE = os.path.join(os.path.dirname(__file__), "nb_aio_presets.json")

# Paramètres sauvegardés (jamais les clés API ni les images)
_PRESET_KEYS = [
    "preset_name",
    "provider",
    "prompt",
    "image_size",
    "model",
    "batch_size",
    "use_search",
    "system_instructions",
    "video_duration",
    "aspect_ratio",
    "temperature",
    "top_p",
    "fal_safety_tolerance",
    "fal_enable_web_search",
    "disable_safety_threshold",
    "video_mode_enabled",
]

def _load_presets() -> dict:
    """Charge les presets depuis le fichier JSON."""
    if not os.path.isfile(_PRESETS_FILE):
        return {}
    try:
        with open(_PRESETS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}
    except Exception as e:
        print(f"⚠️  [Presets] Unable to load : {e}")
        return {}

def _save_presets(presets: dict) -> bool:
    """Sauvegarde les presets dans le fichier JSON."""
    try:
        data = {str(k): v for k, v in presets.items()}
        with open(_PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ [Presets] Saved → {_PRESETS_FILE}")
        return True
    except Exception as e:
        print(f"❌ [Presets] Error sauvegarde : {e}")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Endpoint REST  /nb_aio/save_preset   (appelé par le JS du bouton)
# /nb_aio/load_presets                 (appelé au chargement du node)
#
# Guard sys.modules : évite le double-enregistrement si une copie backup
# du fichier (.py) est présente dans le même dossier custom_nodes.
# ─────────────────────────────────────────────────────────────────────────────
if PromptServer and PromptServer.instance and not getattr(PromptServer.instance, "_nb_aio_routes_registered", False):
    PromptServer.instance._nb_aio_routes_registered = True

    @PromptServer.instance.routes.post("/nb_aio/save_preset")
    async def api_save_preset(request):
        try:
            body     = await request.json()
            slot     = int(body.get("slot", 1))
            params   = body.get("params", {})

            if slot < 1 or slot > 10:
                return web.json_response(
                    {"success": False, "error": "Invalid slot (1-10)"},
                    status=400,
                )

            # Filtrer uniquement les clés autorisées
            filtered = {k: v for k, v in params.items() if k in _PRESET_KEYS}

            presets = _load_presets()
            presets[slot] = filtered
            ok = _save_presets(presets)

            if ok:
                print(f"💾 [Presets] Slot {slot} saved via JS button")
                return web.json_response({
                    "success": True,
                    "slot":    slot,
                    "params":  filtered,
                })
            else:
                return web.json_response(
                    {"success": False, "error": "File write failed"},
                    status=500,
                )
        except Exception as e:
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @PromptServer.instance.routes.get("/nb_aio/load_presets")
    async def api_load_presets(request):
        try:
            presets = _load_presets()
            # Convertit les clés int → str pour JSON
            data = {str(k): v for k, v in presets.items()}
            return web.json_response({"success": True, "presets": data})
        except Exception as e:
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )
else:
    print("⚠️ [NanaBanana] Routes /nb_aio registration skipped (running standalone or already registered).")

# ─────────────────────────────────────────────────────────────────────────────
# Auto-install fal-client
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_fal_client() -> bool:
    try:
        import fal_client  # noqa: F401
        # Supprime le spam console des requêtes HTTP internes de fal_client (httpx)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        return True
    except ImportError:
        print("📦 [Fal.ai] fal-client not found — installing...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "fal-client", "--quiet"
            ])
            logging.getLogger("httpx").setLevel(logging.WARNING)
            print("✅ [Fal.ai] fal-client installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ [Fal.ai] fal-client install failed: {e}")
            return False

# ─────────────────────────────────────────────────────────────────────────────
# Auto-install google-auth
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_google_auth() -> bool:
    try:
        import google.oauth2.service_account
        import google.auth.transport.requests
        return True
    except ImportError:
        print("📦 [Vertex AI] google-auth not found — installing...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "google-auth", "google-auth-httplib2", "--quiet"
            ])
            print("✅ [Vertex AI] google-auth installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ [Vertex AI] google-auth install failed: {e}")
            return False

# ─────────────────────────────────────────────────────────────────────────────
# Helpers REST
# ─────────────────────────────────────────────────────────────────────────────
def _tensor_to_base64_part(img_tensor) -> dict:
    pil_img  = tensor_to_pil(img_tensor)
    buf      = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"inline_data": {"mime_type": "image/png", "data": b64_data}}

def _extract_image_bytes_from_nb2_response(data: dict) -> bytes | None:
    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline:
                raw = inline.get("data", "")
                if raw:
                    return base64.b64decode(raw)
    return None

def _fal_short_error(e: Exception, max_len: int = 220) -> str:
    """Extrait un message d'erreur court depuis une exception Fal.ai.
    FalClientHTTPError includes the full payload (prompt, images…) — we extract
    just the 'msg' field if present, otherwise truncate to max_len characters."""
    import re
    err_str = str(e)
    if len(err_str) <= max_len:
        return err_str
    # Essai d'extraction du champ 'msg' dans l'erreur JSON structurée
    m = re.search(r"'msg':\s*'([^']{1,300})'", err_str)
    if m:
        short = m.group(1)
        t = re.search(r"'type':\s*'([^']+)'", err_str)
        return f"{t.group(1)}: {short}" if t else short
    return err_str[:max_len] + " …[truncated]"

def _tensor_to_fal_url(img_tensor, fal_client, tag: str = "", idx: int = 1) -> str:
    pil_img   = tensor_to_pil(img_tensor)
    buf       = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    url       = fal_client.upload(img_bytes, content_type="image/png")
    print(f"✅ {tag} Image {idx} uploaded → {url}")
    return url

# ─────────────────────────────────────────────────────────────────────────────
# Vertex AI — credentials
# ─────────────────────────────────────────────────────────────────────────────
def _load_vertex_credentials(json_path: str):
    if not _ensure_google_auth():
        raise RuntimeError("❌ Unable to install google-auth.")

    from google.oauth2 import service_account

    json_path = json_path.strip()
    if not json_path:
        raise ValueError("❌ Vertex AI JSON file path is empty.")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"❌ Vertex AI JSON file not found: {json_path}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            sa_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ Invalid JSON file: {e}")

    required_fields = ["type", "project_id", "private_key", "client_email"]
    missing = [field for field in required_fields if field not in sa_data]
    if missing:
        raise ValueError(f"❌ Incomplete JSON file — missing fields: {missing}")

    if sa_data.get("type") != "service_account":
        raise ValueError(f"❌ Unsupported credentials type: '{sa_data.get('type')}'")

    scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/generative-language",
    ]
    credentials = service_account.Credentials.from_service_account_file(
        json_path, scopes=scopes
    )
    project_id = sa_data.get("project_id", "")
    print(f"✅ [Vertex AI] Credentials loaded — project : {project_id}")
    return credentials, project_id


def _load_vertex_json_folder(folder_path: str) -> list:
    """Scans a folder and returns the sorted list of .json files (service accounts)."""
    folder_path = folder_path.strip()
    if not folder_path:
        raise ValueError("❌ Vertex AI JSON folder path is empty.")
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"❌ Vertex AI JSON folder not found: {folder_path}")
    json_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".json")
    ])
    if not json_files:
        raise FileNotFoundError(f"❌ No .json file found in: {folder_path}")
    return json_files


def _make_vertex_client_from_json(json_path: str, location: str, model_name: str = "") -> "genai.Client":
    json_path = (json_path or "").strip()
    credentials = None
    project_id = None
    if not json_path:
        import google.auth
        try:
            credentials, project_id = google.auth.default()
            print(f"🌐 [Vertex AI] Implicit ADC Connection loaded — project={project_id}")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load default GCP credentials: {e}")
    else:
        credentials, project_id = _load_vertex_credentials(json_path)

    effective_location = "global"  # gemini-3-pro-image-preview uniquement disponible en global sur Vertex
    print(
        f"🌐 [Vertex AI] Connection — project={project_id} | "
        f"location={effective_location} | model={model_name or '?'}"
    )
    # google-genai >=1.0 attend timeout en MILLISECONDES.
    # On tente d'abord ms (180_000 = 180s), fallback secondes pour anciens SDK.
    http_opts = None
    for _to in (395_000, 395):  # 395s — just under _VERTEX_CALL_TIMEOUT (400s)
        try:
            http_opts = genai.types.HttpOptions(timeout=_to, max_retries=0)
            break
        except Exception:
            continue
            
    client_kwargs = {
        "vertexai": True,
        "project": project_id,
        "location": effective_location,
    }
    if credentials:
        client_kwargs["credentials"] = credentials
    if http_opts:
        client_kwargs["http_options"] = http_opts

    client = genai.Client(**client_kwargs)
    return client

# ─────────────────────────────────────────────────────────────────────────────
# Node principal
# ─────────────────────────────────────────────────────────────────────────────
class NanoBananaAIO:
    _vertex_rotation_offset  = 0
    _failed_upload_services: set = set()  # services that failed this session  # rotation round-robin entre les projets Vertex

    def __init__(self):
        self._preview_warning_shown = False

    @classmethod
    def INPUT_TYPES(cls):
        # On déclare la liste complète (image + vidéo) pour que la
        # validation côté serveur ComfyUI accepte les valeurs
        # "Veo 3", "Veo 3.1", etc. quand le JS bascule en mode vidéo.
        # Le frontend JS réduit l'affichage au sous-ensemble pertinent.
        model_names = list(_MODEL_MAP.keys()) + list(_VIDEO_MODEL_MAP.keys()) + sorted(_KLING_MODELS) + sorted(_SEEDANCE_MODELS)
        return {
            "required": {
                "provider": (["GOOGLE", "WAVESPEED", "KIE", "FAL", "VERTEX"], {"default": "GOOGLE"}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default":   "A futuristic nano banana dish",
                    "tooltip":   "Generation / transformation prompt",
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default":   "",
                    "tooltip":   "Negative prompt — elements to avoid in the image.\n⚠️ Supported by: FAL, WaveSpeed (NB2), KIE.\nIgnored by Google / Vertex (Gemini does not support negative prompts).",
                }),
                # Liste étendue pour inclure les resolutions vidéo
                # ("720p", "1080p") : le JS restreint l'affichage selon
                # le mode actif, mais la validation serveur a besoin de
                # la liste complète.
                "image_size": (["1K", "2K", "4K", "8K", "480p", "720p", "1080p"], {
                    "default": "2K",
                    "tooltip": "Output resolution (image: 1K/2K/4K/8K · video: 480p/720p/1080p)\n8K disponible uniquement sur Nano Banana Pro via WaveSpeed (edit-ultra).",
                }),
            },
            "optional": {
                # ── API KEYS ──────────────────────────────────────────
                "gemini_api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "[GOOGLE] Google AI Studio API key.",
                }),
                "wavespeed_api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "[WAVESPEED] WaveSpeed API key.",
                }),
                "kie_api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "[KIE] Kie.ai API key.",
                }),
                "fal_api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "[FAL] Fal.ai API key.",
                }),
                "vertex_json_folder": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "[VERTEX] Path to the folder containing service account JSON files (1 .json file = 1 project = 1 batch slot).",
                }),
                # vertex_location : hardcodé "us-central1" — masqué du node
                # vertex_gcs_bucket : valeur vide par défaut — masqué du node
                "disable_safety_threshold": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Disable the safety filter.",
                }),
                "model": (model_names, {
                    "default": model_names[0],
                    "tooltip": "Model to use.",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Number of images to generate in parallel",
                }),
                "use_search": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "[GOOGLE / VERTEX] Enable Google Search grounding",
                }),
                "system_instructions": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip":   "[GOOGLE / VERTEX] System instructions",
                }),
                "video_duration": ("INT", {
                    "default": 8,
                    "min": _VIDEO_DURATION_MIN,
                    "max": _VIDEO_DURATION_MAX,
                    "step": 1,
                }),
                "video_generate_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "[Video] Generate audio track. Supported by all Veo providers.",
                }),
                "aspect_ratio": (
                    ["auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
                     "9:16", "16:9", "21:9"],
                    {"default": "auto",
                     "tooltip": "auto = détecte l'aspect ratio des images en input "
                                "et sélectionne le plus proche parmi les ratios supportés. "
                                "Fallback 1:1 si aucune image en input."},
                ),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "[GOOGLE / VERTEX] Model creativity",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "[GOOGLE / VERTEX] Nucleus sampling",
                }),
                "fal_safety_tolerance": (["1", "2", "3", "4", "5", "6"], {
                    "default": "4",
                    "tooltip": "[FAL NB] 1 = strict | 6 = permissive",
                }),
                "fal_enable_web_search": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "[FAL NB] Enable web search",
                }),
                "gpt2_image_quality": (["high", "medium", "low"], {
                    "default": "high",
                    "tooltip": "[GPT Image 2.0 / FAL] Generation quality. Ignored on other models.",
                }),
                "image_1": ("IMAGE", {"tooltip": "Main source image"}),
                "image_2": ("IMAGE", {"tooltip": "Image source 2"}),
                "image_3": ("IMAGE", {"tooltip": "Image source 3"}),
                "image_4": ("IMAGE", {"tooltip": "Image source 4"}),
                "image_5": ("IMAGE", {"tooltip": "Image source 5"}),
                "video_reference": ("AB_VIDEO", {
                    "tooltip": "[Kling 3.0 Motion Control] Connect an 'Aiorbust Video Loader' node.\n"
                               "The duration of the generated video will match the reference.",
                }),
                # ── Video mode controls (managed by JS DOM widget) ───────────────
                "video_mode_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "[Video] Enabled by JS toggle — do not modify manually.",
                }),
                # ── Automation Face Swap controls (managed by JS DOM widget) ──────
                "face_swap_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "[Automation] Enabled by JS block — do not modify manually.",
                }),
                "breast_refiner_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "[Automation] Enabled by JS block — do not modify manually.",
                }),
                "low_neck_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "[Automation] Enabled by JS block — do not modify manually.",
                }),
                "amateur_mode_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "[Automation] Amateur, Low quality input match — overrides the default face swap prompt.",
                }),
                "face_expression": ("STRING", {
                    "default": "Neutral",
                    "tooltip": "[Automation] Facial expression — managed by JS block — do not modify manually.",
                }),
                "face_swap_custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "[Automation] Extra prompt appended to the face swap prompt — managed by JS block — do not modify manually.",
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES  = ("images", "video", "grounding_sources")
    OUTPUT_NODE   = False
    FUNCTION      = "generate_unified"
    CATEGORY      = "Aiorbust/NanoBanana"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Retourne toujours une valeur unique → force la ré-exécution à chaque fois
        # et garantit que la console affiche bien les logs de génération.
        return float("nan")

    # ─────────────────────────────────────────────────────────────
    #  Point d'entrée principal
    # ─────────────────────────────────────────────────────────────
    def generate_unified(
        self,
        provider        = "GOOGLE",
        prompt          = "",
        negative_prompt = "",
        image_size      = "2K",
        gemini_api_key    = "",
        wavespeed_api_key = "",
        kie_api_key       = "",
        fal_api_key       = "",
        vertex_json_folder = "",
        vertex_location   = "us-central1",
        vertex_gcs_bucket = "",
        disable_safety_threshold = False,
        model             = "Nano Banana Pro",
        batch_size        = 1,
        use_search        = True,
        system_instructions = None,
        aspect_ratio      = "1:1",
        temperature       = 1.0,
        top_p             = 0.95,
        fal_safety_tolerance  = "4",
        fal_enable_web_search = False,
        image_1 = None, image_2 = None, image_3 = None,
        image_4 = None, image_5 = None,
        # ── Video mode ───────────────────────────────────────────────
        video_mode_enabled    = False,
        video_duration        = 8,
        video_generate_audio  = True,
        # ── Automation Face Swap ─────────────────────────────────────
        face_swap_enabled    = False,
        breast_refiner_enabled = False,
        low_neck_enabled       = False,
        amateur_mode_enabled   = False,
        face_expression        = "Neutral",
        face_swap_custom_prompt = "",
        vertex_project_id      = "",
        vertex_location_legacy = "",
        gpt2_image_quality     = "high",
        video_reference        = None,
    ):
        try:
            # ── Suite normale ─────────────────────────────────────
            if (not prompt or not prompt.strip()) and not face_swap_enabled:
                return self._handle_error("Prompt cannot be empty.")

            g_key   = (gemini_api_key    or "").strip()
            ws_key  = (wavespeed_api_key or "").strip()
            kie_key = (kie_api_key       or "").strip()
            fal_key = (fal_api_key       or "").strip()
            vj_folder = (vertex_json_folder or "").strip()
            v_loc   = "us-central1"                          # hardcodé — widget masqué
            v_gcs   = (vertex_gcs_bucket or "").strip()

            # ── VIDEO MODE ────────────────────────────────────────
            if video_mode_enabled:
                input_images_vid = [
                    img for img in [image_1, image_2, image_3, image_4, image_5]
                    if img is not None
                ]
                # Clamp durée selon le model
                _dur = int(video_duration)
                if model == "Kling 2.6":
                    _dur = 10 if _dur > 7 else 5          # uniquement 5 ou 10
                elif model == "Kling 3.0":
                    _dur = max(3, min(15, _dur))           # 3–15 s
                elif model == "Kling 3.0 Motion Control":
                    _dur = 0                               # durée = vidéo de référence
                elif model == "Seedance 2.0":
                    _dur = max(4, min(15, _dur))           # 4–15 s
                else:
                    _dur = max(4, min(8, _dur))            # Veo : 4–8 s
                return self._generate_video(
                    provider             = provider,
                    prompt               = prompt,
                    model                = model,
                    video_resolution     = image_size,
                    aspect_ratio         = aspect_ratio,
                    duration             = _dur,
                    generate_audio       = bool(video_generate_audio),
                    gemini_api_key       = g_key,
                    wavespeed_api_key    = ws_key,
                    kie_api_key          = kie_key,
                    fal_api_key          = fal_key,
                    fal_safety_tolerance = fal_safety_tolerance,
                    vertex_json_folder   = vj_folder,
                    vertex_location      = v_loc,
                    vertex_gcs_bucket    = v_gcs,
                    image_tensors        = input_images_vid,
                    video_reference      = video_reference,
                    disable_safety       = disable_safety_threshold,
                )
            # ─────────────────────────────────────────────────────


            actual_model     = _MODEL_MAP.get(model, "gemini-3-pro-image-preview")
            is_nb2           = (model == "Nano Banana 2")
            is_seedream      = (model == "Seedream 4.5")
            is_gpt2          = (model == "GPT Image 2.0")

            # GPT Image 2.0 — disponible sur WAVESPEED, KIE et FAL uniquement
            if is_gpt2 and provider not in ("WAVESPEED", "KIE", "FAL"):
                return self._handle_error(
                    "❌ GPT Image 2.0 is only available via WAVESPEED, KIE or FAL.\n"
                    "→ Select one of these three providers."
                )
            # Sécurité aspect_ratio pour GPT Image 2.0 : fallback vers le plus proche supporté
            if is_gpt2 and aspect_ratio not in _GPT2_SUPPORTED_AR:
                fallback = _GPT2_NEAREST_AR.get(aspect_ratio, "1:1")
                print(f"⚠️  [GPT2] Aspect ratio {aspect_ratio!r} not supported → fallback {fallback!r}")
                aspect_ratio = fallback
            safety_threshold = "OFF" if disable_safety_threshold else None

            input_images = [
                img for img in [image_1, image_2, image_3, image_4, image_5]
                if img is not None
            ]

            # ── AUTO ASPECT RATIO ─────────────────────────────────────────
            if aspect_ratio == "auto":
                _AR_SUPPORTED = [
                    ("1:1",  1/1),  ("2:3",  2/3),  ("3:2",  3/2),
                    ("3:4",  3/4),  ("4:3",  4/3),  ("4:5",  4/5),
                    ("5:4",  5/4),  ("9:16", 9/16), ("16:9", 16/9),
                    ("21:9", 21/9),
                ]
                if input_images:
                    _ratios = []
                    for _t in input_images:
                        # tensor shape: [B, H, W, C] or [H, W, C]
                        _sh = _t.shape
                        _h, _w = (_sh[-3], _sh[-2]) if len(_sh) == 4 else (_sh[0], _sh[1])
                        if _h > 0 and _w > 0:
                            _ratios.append(_w / _h)
                    if _ratios:
                        _avg = sum(_ratios) / len(_ratios)
                        aspect_ratio = min(_AR_SUPPORTED, key=lambda x: abs(x[1] - _avg))[0]
                        print(f"🔲 [Auto AR] avg ratio={_avg:.3f} → {aspect_ratio}")
                    else:
                        aspect_ratio = "1:1"
                        print("🔲 [Auto AR] could not read image dims → fallback 1:1")
                else:
                    aspect_ratio = "1:1"
                    print("🔲 [Auto AR] no input images → fallback 1:1")

            # ── FACE SWAP AUTOMATION ─────────────────────────────────────
            if face_swap_enabled:
                return self._run_face_swap_automation(
                    provider               = provider,
                    model                  = model,
                    image_size             = image_size,
                    aspect_ratio           = aspect_ratio,
                    batch_size             = batch_size,
                    disable_safety         = disable_safety_threshold,
                    breast_refiner_enabled = breast_refiner_enabled,
                    low_neck_enabled       = low_neck_enabled,
                    amateur_mode_enabled   = amateur_mode_enabled,
                    face_expression        = face_expression,
                    face_swap_custom_prompt = face_swap_custom_prompt,
                    image_1                = image_1,
                    image_2                = image_2,
                    gemini_api_key         = g_key,
                    wavespeed_api_key      = ws_key,
                    kie_api_key            = kie_key,
                    fal_api_key            = fal_key,
                    vertex_json_folder      = vj_folder,
                    vertex_location        = v_loc,
                    temperature            = temperature,
                    top_p                  = top_p,
                    fal_safety_tolerance   = fal_safety_tolerance,
                )
            # ─────────────────────────────────────────────────────────────────

            # ── Garde-fous Seedream ───────────────────────────────
            if is_seedream and provider in ("GOOGLE", "VERTEX"):
                return self._handle_error(
                    f"❌ Seedream 4.5 is not available with provider {provider}.\n"
                    "Select WAVESPEED, KIE or FAL."
                )
            if is_seedream and not input_images:
                return self._handle_error(
                    "❌ Seedream 4.5 requires at least one input image."
                )

            # ── WAVESPEED ─────────────────────────────────────────
            if provider == "WAVESPEED":
                if is_seedream:
                    return self._generate_seedream_wavespeed(
                        prompt, input_images, image_size, aspect_ratio,
                        batch_size=batch_size, ws_api_key=ws_key,
                        disable_safety=disable_safety_threshold,
                    )
                if is_gpt2:
                    return self._generate_gpt2_wavespeed(
                        prompt, input_images, aspect_ratio,
                        image_size=image_size,
                        batch_size=batch_size, ws_api_key=ws_key,
                    )
                print("ℹ️  [WaveSpeed] safety ignored.")
                return self._generate_wavespeed(
                    prompt, input_images, image_size, "jpeg",
                    aspect_ratio, batch_size=batch_size,
                    ws_api_key=ws_key, is_nb2=is_nb2,
                    negative_prompt=negative_prompt,
                )

            # ── KIE ───────────────────────────────────────────────
            if provider == "KIE":
                if is_seedream:
                    return self._generate_seedream_kie(
                        prompt, input_images, image_size, aspect_ratio,
                        kie_api_key=kie_key, ws_api_key=ws_key,
                        batch_size=batch_size,
                        disable_safety=disable_safety_threshold,
                    )
                if is_gpt2:
                    return self._generate_gpt2_kie(
                        prompt, input_images, aspect_ratio,
                        image_size=image_size,
                        kie_api_key=kie_key, ws_api_key=ws_key,
                        batch_size=batch_size,
                        disable_safety=disable_safety_threshold,
                    )
                print("ℹ️  [Kie.ai] safety ignored.")
                return self._generate_kie(
                    prompt, input_images, image_size, aspect_ratio,
                    kie_api_key=kie_key, ws_api_key=ws_key,
                    batch_size=batch_size, is_nb2=is_nb2,
                    negative_prompt=negative_prompt,
                )

            # ── FAL ───────────────────────────────────────────────
            if provider == "FAL":
                if is_seedream:
                    return self._generate_seedream_fal(
                        prompt=prompt,
                        image_tensors=input_images,
                        image_size=image_size,
                        aspect_ratio=aspect_ratio,
                        batch_size=batch_size,
                        fal_api_key=fal_key,
                        disable_safety=disable_safety_threshold,
                    )
                if is_gpt2:
                    return self._generate_gpt2_fal(
                        prompt=prompt,
                        image_tensors=input_images,
                        aspect_ratio=aspect_ratio,
                        image_size=image_size,
                        gpt2_quality=gpt2_image_quality,
                        batch_size=batch_size,
                        fal_api_key=fal_key,
                    )
                return self._generate_fal(
                    prompt=prompt,
                    image_tensors=input_images,
                    image_size=image_size,
                    aspect_ratio=aspect_ratio,
                    batch_size=batch_size,
                    fal_api_key=fal_key,
                    is_nb2=is_nb2,
                    safety_tolerance=fal_safety_tolerance,
                    enable_web_search=fal_enable_web_search,
                    negative_prompt=negative_prompt,
                )

            # ── VERTEX ────────────────────────────────────────────
            if provider == "VERTEX":
                vj_files = []
                if vj_folder:
                    try:
                        vj_files = _load_vertex_json_folder(vj_folder)
                    except Exception as _e:
                        return self._handle_error(str(_e))
                    print(f"📁 [Vertex] {len(vj_files)} project(s) detected")
                    if batch_size > len(vj_files):
                        print(f"⚠️  [Vertex] batch_size={batch_size} > projects={len(vj_files)} → batch reduced to {len(vj_files)}")
                        batch_size = len(vj_files)
                    # Rotation : décale la liste pour répartir les quotas entre appels
                    _off = NanoBananaAIO._vertex_rotation_offset % len(vj_files)
                    vj_files = vj_files[_off:] + vj_files[:_off]
                    NanoBananaAIO._vertex_rotation_offset = (_off + batch_size) % len(vj_files)
                    print(f"🔄 [Vertex] Rotation offset={_off} → projects used : {[os.path.basename(f) for f in vj_files[:batch_size]]}")
                else:
                    print("🌐 [Vertex] Run without rotation json files (using implicit ADC credentials)")
                if is_nb2:
                    return self._generate_nb2_vertex(
                        prompt=prompt,
                        image_tensors=input_images,
                        image_size=image_size,
                        aspect_ratio=aspect_ratio,
                        temperature=temperature,
                        vertex_json_files=vj_files,
                        vertex_location=v_loc,
                        batch_size=batch_size,
                        disable_safety=disable_safety_threshold,
                    )
                vertex_safety_threshold = "BLOCK_NONE" if disable_safety_threshold else None
                return self._generate_vertex(
                    prompt=prompt,
                    image_tensors=input_images,
                    image_size=image_size,
                    aspect_ratio=aspect_ratio,
                    temperature=temperature,
                    top_p=top_p,
                    use_search=use_search,
                    system_instructions=system_instructions,
                    batch_size=batch_size,
                    vertex_json_files=vj_files,
                    vertex_location=v_loc,
                    model_name=actual_model,
                    safety_threshold=vertex_safety_threshold,
                )

            # ── GOOGLE ────────────────────────────────────────────
            if is_nb2:
                if not g_key:
                    return self._handle_error(
                        "❌ gemini_api_key missing for Nano Banana 2 (GOOGLE)."
                    )
                return self._generate_nb2_google(
                    prompt=prompt,
                    image_tensors=input_images,
                    image_size=image_size,
                    aspect_ratio=aspect_ratio,
                    temperature=temperature,
                    g_key=g_key,
                    batch_size=batch_size,
                    disable_safety=disable_safety_threshold,
                )

            approach   = self._detect_approach(g_key, "", "")
            model_name = actual_model

            if not (1 <= batch_size <= 10):
                return self._handle_error("batch_size must be between 1 and 10.")

            valid_ratios = [
                "1:1", "2:3", "3:2", "3:4", "4:3",
                "4:5", "5:4", "9:16", "16:9", "21:9",
            ]
            if aspect_ratio not in valid_ratios:
                return self._handle_error(f"Invalid ratio: {aspect_ratio}")
            if image_size not in ("1K", "2K", "4K", "8K"):
                return self._handle_error(f"Invalid image_size: {image_size}")

            contents = [prompt]
            for i, img_tensor in enumerate(input_images, 1):
                print(f"📷 Image {i} added to Google context...")
                contents.append(tensor_to_pil(img_tensor))

            common = dict(
                g_key=g_key, v_proj="", v_loc="",
                safety_threshold=safety_threshold,
            )

            if batch_size == 1:
                return self._generate_single_image(
                    model_name, prompt, use_search, approach, contents,
                    aspect_ratio, image_size, temperature, top_p,
                    system_instructions, **common
                )
            else:
                return self._generate_batch_parallel(
                    model_name, prompt, batch_size, use_search, approach, contents,
                    aspect_ratio, image_size, temperature, top_p,
                    system_instructions, **common
                )

        except ValueError as e:
            return self._handle_error(f"ValueError : {e}")
        except TypeError as e:
            return self._handle_error(f"TypeError : {e}")
        except Exception as e:
            return self._handle_error(f"{type(e).__name__} : {e}")

    # ─────────────────────────────────────────────────────────────
    #  _handle_error
    # ─────────────────────────────────────────────────────────────
    def _handle_error(self, message: str, batch_idx=None):
        if batch_idx is not None:
            raise RuntimeError(message)
        print(f"\033[91mERROR: {message}\033[0m")
        return (torch.zeros(1, 64, 64, 3), "", "")




    # ─────────────────────────────────────────────────────────────
    #  VERTEX AI — Nano Banana Pro (SDK genai)
    # ─────────────────────────────────────────────────────────────
    def _generate_vertex(
        self,
        prompt, image_tensors, image_size, aspect_ratio,
        temperature=1.0, top_p=0.95,
        use_search=True, system_instructions=None,
        batch_size=1,
        vertex_json_path="", vertex_location="us-central1",
        model_name="gemini-3-pro-image-preview",
        safety_threshold=None,
        _batch_idx=None,
        vertex_json_files=None,
    ):
        # Résolution des fichiers JSON : liste (multi-projets) ou chemin unique
        _files = vertex_json_files if vertex_json_files else ([vertex_json_path] if vertex_json_path else [])
        if not _files:
            _files = [""]

        if batch_size and batch_size > 1:
            print(f"\n🔥 Parallel generation of {batch_size} images (Vertex AI)...\n")
            generated_images      = []
            all_text_responses    = []
            all_grounding_sources = []
            errors                = []

            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {
                    executor.submit(
                        self._generate_vertex,
                        prompt, image_tensors, image_size, aspect_ratio,
                        temperature, top_p, use_search, system_instructions,
                        1, _files[i % len(_files)], vertex_location, model_name,
                        safety_threshold, i + 1,
                    ): i for i in range(batch_size)
                }
                _FuturesTE = __import__('concurrent.futures', fromlist=['TimeoutError']).TimeoutError
                _NBP_OUT = 430 if str(image_size).upper() in ("4K", "8K") else 280
                try:
                    _completed = as_completed(futures, timeout=_NBP_OUT)
                except Exception:
                    _completed = as_completed(futures)
                _processed = set()
                try:
                    for future in _completed:
                        _processed.add(future)
                        idx = futures[future] + 1
                        try:
                            result = future.result(timeout=5)
                            generated_images.append(result[0])
                            all_text_responses.append(result[1])
                            all_grounding_sources.append(result[2])
                        except _FuturesTE:
                            errors.append(f"Batch {idx}: timeout")
                            print(f"❌ [Vertex Batch {idx}] Timeout — future still running")
                        except Exception as e:
                            errors.append(f"Batch {idx}: {e}")
                            print(f"❌ [Vertex Batch {idx}] Failed — {e}")
                except _FuturesTE:
                    # as_completed outer timeout fired — salvage any completed futures
                    print(f"⚠️  [Vertex AI] as_completed timeout — collecting results from finished futures...")
                    for _f, _i in futures.items():
                        if _f in _processed:
                            continue
                        _fidx = _i + 1
                        if _f.done():
                            try:
                                _r = _f.result(timeout=0)
                                generated_images.append(_r[0])
                                all_text_responses.append(_r[1])
                                all_grounding_sources.append(_r[2])
                                print(f"✅ [Vertex Batch {_fidx}] Recovered from completed future")
                            except Exception as _fe:
                                errors.append(f"Batch {_fidx}: {_fe}")
                                print(f"❌ [Vertex Batch {_fidx}] Failed — {_fe}")
                        else:
                            errors.append(f"Batch {_fidx}: outer timeout (still running)")
                            print(f"❌ [Vertex Batch {_fidx}] Outer timeout — still running")

            print(f"\n✨ Vertex AI — {len(generated_images)}/{batch_size} successful\n")
            if errors:
                for err in errors:
                    print(f"   ⚠️  {err}")
            if not generated_images:
                return self._handle_error(
                    f"All Vertex AI generations failed : {'; '.join(errors)}"
                )
            ref_shape    = generated_images[0].shape
            valid_images = [img for img in generated_images if img.shape == ref_shape]
            if not valid_images:
                return self._handle_error("No valid image.")
            combined           = torch.cat(valid_images, dim=0)
            combined_text      = "\n\n".join(all_text_responses)
            combined_grounding = "\n\n".join(all_grounding_sources)
            if errors:
                combined_text = f"⚠️ {len(errors)} image(s) failed\n" + combined_text
            return (combined, combined_text, combined_grounding)

        tag             = f"[Vertex Batch {_batch_idx}]" if _batch_idx else "[Vertex AI]"
        vertex_model_id = "gemini-3-pro-image-preview"
        print(f"ℹ️  [Vertex AI] Model: {vertex_model_id}")

        try:
            client = _make_vertex_client_from_json(
                _files[0], vertex_location, vertex_model_id
            )
        except Exception as e:
            return self._handle_error(
                f"❌ {tag} Unable to create Vertex AI client: {e}", _batch_idx
            )

        config = self._create_config(
            aspect_ratio, image_size, temperature, top_p,
            use_search, vertex_model_id, system_instructions, safety_threshold,
        )

        contents = [prompt]
        for i, img_tensor in enumerate(image_tensors, 1):
            print(f"📷 {tag} Image {i} added to context...")
            contents.append(tensor_to_pil(img_tensor))

        # Fix 3: non-streaming blocking call — more reliable than streaming for
        # thinking models (avoids inter-chunk timeout race on thinking text chunk).
        _VERTEX_CALL_TIMEOUT = 300  # hard cutoff total (HTTP-level + outer safety)
        print(f"\U0001f680 {tag} Generating (model={vertex_model_id})...")

        def _gen_worker():
            return client.models.generate_content(
                model=vertex_model_id, contents=contents, config=config
            )

        _tex  = ThreadPoolExecutor(max_workers=1)
        _fut  = _tex.submit(_gen_worker)

        _response    = None
        _gen_error   = None
        with _Heartbeat(f"{tag} In progress"):
            try:
                _response = _fut.result(timeout=_VERTEX_CALL_TIMEOUT)
            except Exception as _ge:
                _gen_error = _ge

        _tex.shutdown(wait=False)

        image_bytes   = None
        text_response = ""

        if _response and not isinstance(_response, Exception):
            _cands = getattr(_response, "candidates", None) or []
            for _cand in _cands:
                _fr = getattr(_cand, "finish_reason", None)
                if _fr and _fr not in (types.FinishReason.STOP,) \
                        and str(_fr) not in ("0", "FINISH_REASON_UNSPECIFIED", "None"):
                    return self._handle_error(
                        f"\u274c {tag} Generation blocked: finish_reason={_fr}", _batch_idx
                    )
                _content = getattr(_cand, "content", None)
                if _content and getattr(_content, "parts", None):
                    for _part in _content.parts:
                        if getattr(_part, "inline_data", None) and image_bytes is None:
                            image_bytes = _part.inline_data.data
                        elif getattr(_part, "text", None):
                            text_response += _part.text

        grounding_sources = self.extract_grounding_data(_response) if _response else ""

        # ── Diagnostic Vertex (lecture prompt_feedback / usage_metadata) ──
        # But : transformer les "no image data" silencieux en messages
        # explicites avec la raison du blocage cote serveur.
        _pf_bits = []
        _meta_bits = []
        if _response and not isinstance(_response, Exception):
            try:
                _pf = getattr(_response, "prompt_feedback", None) \
                    or getattr(_response, "promptFeedback", None)
                if _pf:
                    _br  = getattr(_pf, "block_reason", None) \
                         or getattr(_pf, "blockReason", None)
                    _brm = getattr(_pf, "block_reason_message", None) \
                         or getattr(_pf, "blockReasonMessage", None)
                    _sr  = getattr(_pf, "safety_ratings", None) \
                         or getattr(_pf, "safetyRatings", None) or []
                    if _br:
                        _pf_bits.append(f"block_reason={_br}")
                    if _brm:
                        _pf_bits.append(f"block_message={_brm!r}")
                    _blocked_cats = []
                    for _r in _sr:
                        _blk = getattr(_r, "blocked", False)
                        _cat = getattr(_r, "category", None)
                        _prb = getattr(_r, "probability", None)
                        if _blk:
                            _blocked_cats.append(f"{_cat}:{_prb}")
                    if _blocked_cats:
                        _pf_bits.append(f"blocked_categories=[{', '.join(map(str, _blocked_cats))}]")
            except Exception as _e_pf:
                _pf_bits.append(f"pf_parse_err={_e_pf}")
            try:
                _um = getattr(_response, "usage_metadata", None) \
                    or getattr(_response, "usageMetadata", None)
                if _um:
                    _ptc = getattr(_um, "prompt_token_count", None) \
                         or getattr(_um, "promptTokenCount", None)
                    _ctc = getattr(_um, "candidates_token_count", None) \
                         or getattr(_um, "candidatesTokenCount", None)
                    _meta_bits.append(f"tokens=in:{_ptc}/out:{_ctc}")
            except Exception:
                pass
            try:
                _cands_count = len(getattr(_response, "candidates", None) or [])
                _meta_bits.append(f"candidates_count={_cands_count}")
            except Exception:
                pass

        # ── Diagnostic erreur SDK (headers HTTP, request_id, body) ────────
        _err_bits = []
        if _gen_error is not None:
            try:
                _resp = getattr(_gen_error, "response", None)
                if _resp is not None:
                    _hdrs = dict(getattr(_resp, "headers", {}) or {})
                    for _k in ("x-request-id", "X-Request-Id", "x-goog-request-id",
                               "x-ratelimit-remaining", "x-ratelimit-reset"):
                        _v = _hdrs.get(_k)
                        if _v:
                            _err_bits.append(f"{_k}={_v}")
                    _body = getattr(_resp, "text", None)
                    if _body and isinstance(_body, str):
                        _bsnip = _body[:300]
                        _err_bits.append(f"body_snippet={_bsnip!r}")
                _code = getattr(_gen_error, "code", None) \
                      or getattr(_gen_error, "status_code", None)
                if _code:
                    _err_bits.append(f"http_code={_code}")
            except Exception:
                pass

        if image_bytes is None:
            txt_preview = (text_response[:300] + "\u2026") if len(text_response) > 300 else text_response
            if txt_preview:
                print(f"\u26a0\ufe0f  {tag} Model text-only response: {txt_preview!r}")
            err_detail = f" | {type(_gen_error).__name__}: {_gen_error}" if _gen_error else ""
            if _gen_error:
                print(f"\u26a0\ufe0f  {tag} Generation error: {_gen_error}")
            # Logs diagnostic — toujours imprimer ce qu'on a, meme si vide
            if _pf_bits:
                print(f"\U0001f50d {tag} prompt_feedback: {' | '.join(_pf_bits)}")
            if _meta_bits:
                print(f"\U0001f50d {tag} response_meta: {' | '.join(_meta_bits)}")
            if _err_bits:
                print(f"\U0001f50d {tag} sdk_error_detail: {' | '.join(_err_bits)}")
            # Concatenation dans le message d erreur final (visible dans les logs batch)
            _diag_suffix = ""
            if _pf_bits:
                _diag_suffix += " | " + ", ".join(_pf_bits)
            if _err_bits and not _gen_error:
                _diag_suffix += " | " + ", ".join(_err_bits)
            return self._handle_error(
                f"\u274c {tag} No image data in Vertex AI response{err_detail}{_diag_suffix}.", _batch_idx
            )

        pil_image    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np     = np.array(pil_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        print(f"✅ {tag} Generation successful!")
        return (image_tensor, text_response, grounding_sources)

    # ─────────────────────────────────────────────────────────────
    #  VERTEX AI — Nano Banana 2 (REST OAuth2)
    # ─────────────────────────────────────────────────────────────
    def _generate_nb2_vertex(
        self,
        prompt, image_tensors, image_size, aspect_ratio,
        temperature=1.0,
        vertex_json_path="", vertex_location="us-central1",
        batch_size=1, disable_safety=False,
        _batch_idx=None,
        vertex_json_files=None,
    ):
        # Résolution des fichiers JSON
        _files = vertex_json_files if vertex_json_files else ([vertex_json_path] if vertex_json_path else [])
        if not _files:
            return self._handle_error("❌ vertex_json missing (NB2).", _batch_idx)

        if batch_size and batch_size > 1:
            print(f"\n🔥 Parallel generation of {batch_size} images (Vertex AI NB2)...\n")
            generated_images   = []
            all_text_responses = []
            errors             = []

            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {
                    executor.submit(
                        self._generate_nb2_vertex,
                        prompt, image_tensors, image_size, aspect_ratio,
                        temperature, _files[i % len(_files)], vertex_location,
                        1, disable_safety, i + 1,
                    ): i for i in range(batch_size)
                }
                _FTE2 = __import__('concurrent.futures', fromlist=['TimeoutError']).TimeoutError
                _proc2 = set()
                _NB2_OUT = 430 if str(image_size).upper() in ("4K", "8K") else (NB2_TIMEOUT_S + 30)
                try:
                    for future in as_completed(futures, timeout=_NB2_OUT):
                        _proc2.add(future)
                        idx = futures[future] + 1
                        try:
                            result = future.result(timeout=10)
                            generated_images.append(result[0])
                            all_text_responses.append(result[1])
                        except Exception as e:
                            errors.append(f"Batch {idx}: {e}")
                            print(f"❌ [Vertex NB2 Batch {idx}] Failed — {e}")
                except _FTE2:
                    print("[Vertex NB2] timeout — salvaging completed futures...")
                    for _f, _i in futures.items():
                        if _f in _proc2: continue
                        _fi = _i + 1
                        if _f.done():
                            try:
                                _r = _f.result(timeout=0)
                                generated_images.append(_r[0])
                                all_text_responses.append(_r[1])
                                print(f"✅ [Vertex NB2 Batch {_fi}] Recovered")
                            except Exception as _fe:
                                errors.append(f"Batch {_fi}: {_fe}")
                        else:
                            errors.append(f"Batch {_fi}: outer timeout")

            if not generated_images:
                return self._handle_error(
                    f"All Vertex NB2 generations failed : {'; '.join(errors)}"
                )
            ref_shape    = generated_images[0].shape
            valid_images = [img for img in generated_images if img.shape == ref_shape]
            if not valid_images:
                return self._handle_error("No valid image.")
            combined      = torch.cat(valid_images, dim=0)
            combined_text = "\n\n".join(all_text_responses)
            if errors:
                combined_text = f"⚠️ {len(errors)} error(s)\n" + combined_text
            return (combined, combined_text, "")

        tag = f"[Vertex NB2 Batch {_batch_idx}]" if _batch_idx else "[Vertex AI NB2]"

        try:
            if not _ensure_google_auth():
                return self._handle_error("❌ google-auth not available.", _batch_idx)
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request as GoogleAuthRequest

            credentials, project_id = _load_vertex_credentials(_files[0])
            credentials.refresh(GoogleAuthRequest())
            access_token = credentials.token
            print(f"🔑 {tag} OAuth2 token obtained (project={project_id})")
        except Exception as e:
            return self._handle_error(
                f"❌ {tag} Unable to obtain OAuth2 token: {e}", _batch_idx
            )

        vertex_nb2_model_id = "gemini-3.1-flash-image-preview"
        vertex_url = (
            f"https://aiplatform.googleapis.com/v1/"
            f"projects/{project_id}/locations/global/"
            f"publishers/google/models/{vertex_nb2_model_id}:streamGenerateContent"
        )

        parts = []
        for i, img_tensor in enumerate(image_tensors, 1):
            print(f"📷 {tag} Adding image {i}...")
            parts.append(_tensor_to_base64_part(img_tensor))
        parts.append({"text": prompt})

        image_config = {}
        if aspect_ratio and aspect_ratio != "auto":
            image_config["aspectRatio"] = aspect_ratio
        if image_size:
            image_config["imageSize"] = image_size

        gen_config: dict = {
            "responseModalities": ["TEXT", "IMAGE"],
            "temperature": temperature,
            "thinkingConfig": {"thinkingBudget": 0, "includeThoughts": False},  # disable thinking — prevents safety over-triggering
        }
        if image_config:
            gen_config["imageConfig"] = image_config

        safety_settings = [
            {"category": cat, "threshold": "OFF"} for cat in _HARM_CATEGORIES
        ]

        payload = {
            "contents":          [{"role": "user", "parts": parts}],
            "systemInstruction": {"parts": [{"text": NB2_SYSTEM_PROMPT}]},
            "generationConfig":  gen_config,
            "safetySettings":    safety_settings,
        }
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type":  "application/json",
        }

        import json as _json
        print(f"⏳ {tag} Sending NB2 streaming request (timeout: {NB2_TIMEOUT_S}s)...")
        _hb = _Heartbeat(f"{tag} En waiting for response")
        _hb.__enter__()
        try:
            response = requests.post(
                vertex_url, json=payload, headers=headers,
                timeout=NB2_TIMEOUT_S, stream=True,
            )
            response.raise_for_status()

            all_candidates  = []
            prompt_feedback = {}
            error_chunks    = []
            buffer          = ""
            chunk_count     = 0
            raw_lines_seen  = []   # garde les 5 premières lignes non-vides pour debug

            for raw_chunk in response.iter_lines(chunk_size=8192, decode_unicode=True):
                if not raw_chunk:
                    continue
                if len(raw_lines_seen) < 5:
                    raw_lines_seen.append(raw_chunk[:200])
                # Toujours accumuler dans le buffer (y compris [ et ])
                buffer += raw_chunk + "\n"
                line = raw_chunk.strip().rstrip(",")
                if line in ("[", "]"):
                    continue
                try:
                    chunk = _json.loads(line)
                    buffer = ""  # parse ligne successful, réinitialise le buffer
                except Exception:
                    try:
                        chunk = _json.loads(buffer.strip())
                        buffer = ""
                    except Exception:
                        continue

                # Vertex peut renvoyer la réponse en JSON array pretty-printed.
                # Dans ce cas, `chunk` est une list de dicts : on la normalise.
                items = chunk if isinstance(chunk, list) else [chunk]
                for c in items:
                    if not isinstance(c, dict):
                        continue
                    if "promptFeedback" in c:
                        prompt_feedback = c["promptFeedback"]

                    if "error" in c:
                        error_chunks.append(c["error"])

                    if "candidates" in c:
                        for cand in c["candidates"]:
                            finish = cand.get("finishReason", "")
                            if finish and finish not in ("STOP", "MAX_TOKENS", ""):
                                print(f"⚠️  {tag} finishReason={finish} detected in stream.")
                        all_candidates.extend(c["candidates"])
                        chunk_count += 1
                        if chunk_count == 1:
                            print(f"   {tag} First chunk received, generating...")
                            # On a la réponse — stoppe le heartbeat d'attente.
                            try: _hb.__exit__(None, None, None)
                            except Exception: pass

            # ── Fallback : si le stream était pretty-printed, parser le buffer complet ──
            if not all_candidates and buffer.strip():
                try:
                    full_parsed = _json.loads(buffer.strip())
                    _items = full_parsed if isinstance(full_parsed, list) else [full_parsed]
                    for _c in _items:
                        if not isinstance(_c, dict): continue
                        if "promptFeedback" in _c: prompt_feedback = _c["promptFeedback"]
                        if "error" in _c: error_chunks.append(_c["error"])
                        if "candidates" in _c: all_candidates.extend(_c["candidates"])
                    if all_candidates:
                        print(f"   {tag} Pretty-printed response parsed from complete buffer.")
                except Exception as _pe:
                    print(f"⚠️  {tag} Complete buffer not parseable: {_pe}")

            image_bytes   = None
            text_response = ""
            for cand in all_candidates:
                for part in cand.get("content", {}).get("parts", []):
                    if "inlineData" in part and image_bytes is None:
                        image_bytes = base64.b64decode(part["inlineData"]["data"])
                    elif "text" in part:
                        text_response += part["text"]

            if not image_bytes:
                if error_chunks:
                    err = error_chunks[0]
                    code = err.get("code", "?")
                    msg  = err.get("message", str(err))
                    return self._handle_error(
                        f"❌ {tag} API error: code={code} — {msg}", _batch_idx
                    )
                if all_candidates:
                    finish  = all_candidates[-1].get("finishReason", "?")
                    safety  = all_candidates[-1].get("safetyRatings", [])
                    blocked = [r.get("category", "?") for r in safety if r.get("blocked")]
                    detail  = f" | blocked by: {blocked}" if blocked else ""
                    text_in = (text_response[:200] + "…") if text_response else "(vide)"
                    return self._handle_error(
                        f"❌ {tag} No image generated (finishReason={finish}{detail})\n"
                        f"   Text received: {text_in}", _batch_idx
                    )
                else:
                    # No candidates — blocage silencieux ou réponse vide
                    br  = prompt_feedback.get("blockReason") or prompt_feedback.get("block_reason")
                    brm = prompt_feedback.get("blockReasonMessage", "")
                    if br:
                        detail = f"{br} — {brm}" if brm else br
                    elif prompt_feedback:
                        detail = f"promptFeedback={prompt_feedback}"
                    else:
                        detail = "empty response (0 candidates, no promptFeedback)"
                    print(f"🔍 {tag} First bytes received : {raw_lines_seen}")
                    return self._handle_error(
                        f"❌ {tag} Request blocked or empty response — {detail}", _batch_idx
                    )

            pil_image    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_np     = np.array(pil_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            print(f"✅ {tag} Generation successful!")
            return (image_tensor, text_response, "")

        except requests.exceptions.Timeout:
            return self._handle_error(f"❌ {tag} Timeout after {NB2_TIMEOUT_S}s.", _batch_idx)
        except requests.exceptions.HTTPError as e:
            try:
                body = e.response.json()
            except Exception:
                body = e.response.text
            return self._handle_error(
                f"❌ {tag} HTTP {e.response.status_code} : {body}", _batch_idx
            )
        except requests.exceptions.RequestException as e:
            return self._handle_error(f"❌ {tag} Network error : {e}", _batch_idx)
        finally:
            # Garantit l'arrêt du heartbeat même en cas de return/exception.
            try: _hb.__exit__(None, None, None)
            except Exception: pass

    # ─────────────────────────────────────────────────────────────
    #  SEEDREAM 4.5 — WaveSpeed
    # ─────────────────────────────────────────────────────────────
    def _generate_seedream_wavespeed(
        self,
        prompt, image_tensors, image_size, aspect_ratio,
        batch_size=1, ws_api_key="",
        disable_safety=False, _batch_idx=None,
    ):
        if not ws_api_key:
            return self._handle_error("❌ wavespeed_api_key missing!", _batch_idx)

        if batch_size and batch_size > 1:
            print(f"\n🔥 Parallel generation of {batch_size} images (WaveSpeed Seedream 4.5)...\n")
            generated_images = []
            all_info         = []
            errors           = []
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {}
                for i in range(batch_size):
                    fut = executor.submit(
                        self._generate_seedream_wavespeed,
                        prompt, image_tensors, image_size, aspect_ratio,
                        1, ws_api_key, disable_safety, i + 1,
                    )
                    futures[fut] = i
                for future in as_completed(futures):
                    idx = futures[future] + 1
                    try:
                        result = future.result()
                        generated_images.append(result[0])
                        all_info.append(result[1])
                    except Exception as e:
                        errors.append(f"Batch {idx}: {e}")
                        print(f"❌ [SD Batch {idx}] Failed — {e}")

            if not generated_images:
                return self._handle_error(
                    f"All Seedream WaveSpeed generations failed: {'; '.join(errors)}"
                )
            ref_shape    = generated_images[0].shape
            valid_images = [img for img in generated_images if img.shape == ref_shape]
            if not valid_images:
                return self._handle_error("No valid image.")
            combined      = torch.cat(valid_images, dim=0)
            combined_info = "\n\n".join(all_info)
            if errors:
                combined_info = f"⚠️ {len(errors)} image(s) failed\n" + combined_info
            return (combined, combined_info, "")

        tag          = f"[SD Batch {_batch_idx}]" if _batch_idx else "[Seedream WS]"
        headers_auth = {"Authorization": f"Bearer {ws_api_key}"}

        print(f"⬆️  {tag} Uploading {len(image_tensors)} image(s)...")
        image_urls = []
        for idx, img_tensor in enumerate(image_tensors, 1):
            try:
                pil_image  = tensor_to_pil(img_tensor)
                img_buffer = io.BytesIO()
                pil_image.save(img_buffer, format="PNG")
                img_buffer.seek(0)
                upload_resp = requests.post(
                    WAVESPEED_UPLOAD_URL,
                    headers=headers_auth,
                    files={"file": (f"source_{idx}.png", img_buffer, "image/png")},
                    timeout=60,
                )
                upload_resp.raise_for_status()
                upload_data = upload_resp.json()
                image_url   = upload_data.get("data", {}).get("download_url")
                if not image_url:
                    return self._handle_error(
                        f"❌ {tag} Image {idx} upload failed : {upload_data}", _batch_idx
                    )
                image_urls.append(image_url)
                print(f"✅ {tag} Image {idx}/{len(image_tensors)} uploaded")
            except requests.RequestException as e:
                return self._handle_error(
                    f"❌ {tag} Image upload error {idx} : {e}", _batch_idx
                )

        ws_size = _SEEDREAM_WS_SIZE_MAP.get((aspect_ratio, image_size))
        if not ws_size:
            ws_size = _SEEDREAM_WS_SIZE_MAP.get((aspect_ratio, "2K"), "1920*1920")
            print(f"⚠️  {tag} Size not mapped → fallback {ws_size}")

        payload = {
            "images":                image_urls,
            "prompt":                prompt,
            "size":                  ws_size,
            "enable_sync_mode":      False,
            "enable_base64_output":  False,
            "enable_safety_checker": not disable_safety,
        }
        print(f"🚀 {tag} Submitting Seedream 4.5 (size={ws_size})...")

        try:
            submit_resp = requests.post(
                WAVESPEED_SEEDREAM_URL,
                json=payload,
                headers={**headers_auth, "Content-Type": "application/json"},
                timeout=30,
            )
            submit_resp.raise_for_status()
            submit_data = submit_resp.json()
            task_id = (submit_data.get("data") or {}).get("id") or submit_data.get("id")
            if not task_id:
                return self._handle_error(
                    f"❌ {tag} Submission failed: {submit_data}", _batch_idx
                )
            print(f"🔖 {tag} Task ID : {task_id}")
        except requests.RequestException as e:
            return self._handle_error(f"❌ {tag} Submission error : {e}", _batch_idx)

        poll_url  = WAVESPEED_POLL_URL.format(task_id=task_id)
        elapsed   = 0
        poll_data = {}
        print(f"⏳ {tag} Waiting for result (timeout: {WAVESPEED_TIMEOUT_S}s)...")
        while elapsed < WAVESPEED_TIMEOUT_S:
            time.sleep(WAVESPEED_POLL_DELAY)
            elapsed += WAVESPEED_POLL_DELAY
            try:
                poll_resp = requests.get(poll_url, headers=headers_auth, timeout=15)
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
            except requests.RequestException as e:
                print(f"⚠️  {tag} Polling error ({elapsed}s) : {e}")
                continue

            status = poll_data.get("data", {}).get("status", "")
            if status == "completed":
                outputs = poll_data.get("data", {}).get("outputs", [])
                if not outputs:
                    return self._handle_error(f"❌ {tag} Completed but no output.", _batch_idx)
                output_url = outputs[0]
                print(f"✅ {tag} Image ready! ({elapsed}s)")
                break
            elif status == "failed":
                error_msg = poll_data.get("data", {}).get("error", "Error inconnue")
                return self._handle_error(f"❌ {tag} Generation failed: {error_msg}", _batch_idx)
            else:
                print(f"   {tag} [{elapsed}s/{WAVESPEED_TIMEOUT_S}s] status={status!r}...")
        else:
            try:
                requests.delete(WAVESPEED_CANCEL_URL.format(task_id=task_id), headers=headers_auth, timeout=10)
            except Exception:
                pass
            return self._handle_error(f"❌ {tag} Timeout after {WAVESPEED_TIMEOUT_S}s.", _batch_idx)

        try:
            img_resp         = requests.get(output_url, timeout=60)
            img_resp.raise_for_status()
            pil_result       = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            image_np         = np.array(pil_result).astype(np.float32) / 255.0
            image_tensor_out = torch.from_numpy(image_np)[None,]
            timing_ms = poll_data.get("data", {}).get("timings", {}).get("inference", 0)
            info = (
                f"[WaveSpeed] Model: Seedream 4.5 Edit\n"
                f"Size: {ws_size} | Inference time: {timing_ms}ms | Total: {elapsed}s"
            )
            print(f"🎉 {tag} Done!\n{info}")
            return (image_tensor_out, info, "")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Download error : {e}", _batch_idx)

    # ─────────────────────────────────────────────────────────────
    #  GPT IMAGE 2.0 — WaveSpeed
    # ─────────────────────────────────────────────────────────────
    def _generate_gpt2_wavespeed(
        self,
        prompt, image_tensors, aspect_ratio="1:1",
        image_size="2K",
        batch_size=1, ws_api_key="",
        _batch_idx=None,
    ):
        if not ws_api_key:
            return self._handle_error("❌ wavespeed_api_key missing!", _batch_idx)

        if batch_size and batch_size > 1:
            print(f"\n🔥 Parallel generation of {batch_size} images (WaveSpeed GPT Image 2.0)...\n")
            generated_images = []
            all_info         = []
            errors           = []
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {}
                for i in range(batch_size):
                    fut = executor.submit(
                        self._generate_gpt2_wavespeed,
                        prompt, image_tensors, aspect_ratio,
                        image_size, 1, ws_api_key, i + 1,
                    )
                    futures[fut] = i
                for future in as_completed(futures):
                    idx = futures[future] + 1
                    try:
                        result = future.result()
                        generated_images.append(result[0])
                        all_info.append(result[1])
                    except Exception as e:
                        errors.append(f"Batch {idx}: {e}")
                        print(f"❌ [GPT2 Batch {idx}] Failed — {e}")

            if not generated_images:
                return self._handle_error(
                    f"All GPT Image 2.0 WaveSpeed generations failed : {'; '.join(errors)}"
                )
            ref_shape    = generated_images[0].shape
            valid_images = [img for img in generated_images if img.shape == ref_shape]
            if not valid_images:
                return self._handle_error("No valid image.")
            combined      = torch.cat(valid_images, dim=0)
            combined_info = "\n\n".join(all_info)
            if errors:
                combined_info = f"⚠️ {len(errors)} image(s) failed\n" + combined_info
            return (combined, combined_info, "")

        tag          = f"[GPT2 Batch {_batch_idx}]" if _batch_idx else "[GPT Image 2.0 WS]"
        headers_auth = {"Authorization": f"Bearer {ws_api_key}"}

        # Upload images if provided (optional for GPT Image 2.0)
        image_urls = []
        if image_tensors:
            print(f"⬆️  {tag} Uploading {len(image_tensors)} image(s)...")
            for idx, img_tensor in enumerate(image_tensors, 1):
                try:
                    pil_image  = tensor_to_pil(img_tensor)
                    img_buffer = io.BytesIO()
                    pil_image.save(img_buffer, format="PNG")
                    img_buffer.seek(0)
                    upload_resp = requests.post(
                        WAVESPEED_UPLOAD_URL,
                        headers=headers_auth,
                        files={"file": (f"source_{idx}.png", img_buffer, "image/png")},
                        timeout=60,
                    )
                    upload_resp.raise_for_status()
                    upload_data = upload_resp.json()
                    image_url   = upload_data.get("data", {}).get("download_url")
                    if not image_url:
                        return self._handle_error(
                            f"❌ {tag} Image {idx} upload failed : {upload_data}", _batch_idx
                        )
                    image_urls.append(image_url)
                    print(f"✅ {tag} Image {idx}/{len(image_tensors)} uploaded")
                except requests.RequestException as e:
                    return self._handle_error(
                        f"❌ {tag} Image upload error {idx} : {e}", _batch_idx
                    )

        _ws_resolution = SIZE_TO_RESOLUTION.get(image_size, "2k")  # 1k/2k/4k
        _ws_quality    = "high" if image_size in ("2K", "4K") else "medium"
        payload = {
            "prompt":               prompt,
            "images":               image_urls if image_urls else [None],
            "aspect_ratio":         aspect_ratio,
            "quality":              _ws_quality,
            "resolution":           _ws_resolution,
            "enable_base64_output": False,
            "enable_sync_mode":     False,
        }
        print(f"🚀 {tag} Submitting GPT Image 2.0 (aspect_ratio={aspect_ratio} | resolution={_ws_resolution} | quality={_ws_quality})...")

        try:
            submit_resp = requests.post(
                WAVESPEED_GPT2_URL, json=payload,
                headers={**headers_auth, "Content-Type": "application/json"},
                timeout=30,
            )
            submit_resp.raise_for_status()
            submit_data = submit_resp.json()
            # Vérifier une erreur API-level (HTTP 200 mais corps d'erreur)
            api_err = submit_data.get("error") or submit_data.get("message", "")
            if api_err and str(api_err).lower() not in ("", "ok", "success"):
                return self._handle_error(f"❌ {tag} API error: {api_err}", _batch_idx)
            task_id = (submit_data.get("data") or {}).get("id") or submit_data.get("id")
            if not task_id:
                return self._handle_error(
                    f"❌ {tag} Submission failed: {submit_data}", _batch_idx
                )
            print(f"🔖 {tag} Task ID : {task_id}")
        except requests.RequestException as e:
            return self._handle_error(f"❌ {tag} Submission error : {e}", _batch_idx)

        # Statuts terminaux connus (autre que "completed")
        _FAIL_STATUSES    = {"failed", "error", "cancelled", "rejected", "timeout"}
        # Statuts en cours : on continue de poller
        _PENDING_STATUSES = {"pending", "queued", "processing", "running", "starting"}

        poll_url  = WAVESPEED_POLL_URL.format(task_id=task_id)
        elapsed   = 0
        poll_data = {}
        print(f"⏳ {tag} Waiting for result (timeout: {WAVESPEED_TIMEOUT_S}s)...")
        while elapsed < WAVESPEED_TIMEOUT_S:
            time.sleep(WAVESPEED_POLL_DELAY)
            elapsed += WAVESPEED_POLL_DELAY
            try:
                poll_resp = requests.get(poll_url, headers=headers_auth, timeout=15)
                # Errors HTTP fatales → pas la peine de continuer
                if poll_resp.status_code in (401, 403, 404):
                    return self._handle_error(
                        f"❌ {tag} Error HTTP {poll_resp.status_code} during polling.", _batch_idx
                    )
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
            except requests.RequestException as e:
                print(f"⚠️  {tag} Polling error ({elapsed}s) : {e}")
                continue

            status = poll_data.get("data", {}).get("status", "")
            if status == "completed":
                outputs = poll_data.get("data", {}).get("outputs", [])
                if not outputs:
                    return self._handle_error(f"❌ {tag} Completed but no output.", _batch_idx)
                output_url = outputs[0]
                print(f"✅ {tag} Image ready! ({elapsed}s)")
                break
            elif status in _FAIL_STATUSES:
                error_msg = poll_data.get("data", {}).get("error") or f"Status : {status}"
                return self._handle_error(f"❌ {tag} Generation failed: {error_msg}", _batch_idx)
            elif status and status not in _PENDING_STATUSES:
                # Status inattendu et non-pending → on sort immédiatement
                error_msg = poll_data.get("data", {}).get("error") or f"Status inattendu : {status!r}"
                return self._handle_error(f"❌ {tag} Stopped: {error_msg}", _batch_idx)
            else:
                print(f"   {tag} [{elapsed}s/{WAVESPEED_TIMEOUT_S}s] status={status!r}...")
        else:
            try:
                requests.delete(WAVESPEED_CANCEL_URL.format(task_id=task_id), headers=headers_auth, timeout=10)
            except Exception:
                pass
            return self._handle_error(f"❌ {tag} Timeout after {WAVESPEED_TIMEOUT_S}s.", _batch_idx)

        try:
            img_resp         = requests.get(output_url, timeout=60)
            img_resp.raise_for_status()
            pil_result       = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            image_np         = np.array(pil_result).astype(np.float32) / 255.0
            image_tensor_out = torch.from_numpy(image_np)[None,]
            timing_ms = poll_data.get("data", {}).get("timings", {}).get("inference", 0)
            info = (
                f"[WaveSpeed] Model: GPT Image 2.0\n"
                f"Aspect ratio: {aspect_ratio} | Inference time: {timing_ms}ms | Total: {elapsed}s"
            )
            print(f"🎉 {tag} Done!\n{info}")
            return (image_tensor_out, info, "")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Download error : {e}", _batch_idx)

    # ─────────────────────────────────────────────────────────────
    #  SEEDREAM 4.5 — Kie.ai
    # ─────────────────────────────────────────────────────────────
    def _generate_seedream_kie(
        self,
        prompt, image_tensors, image_size, aspect_ratio,
        kie_api_key="", ws_api_key="",
        batch_size=1, disable_safety=False,
        _batch_idx=None,
    ):
        if not kie_api_key:
            return self._handle_error("❌ kie_api_key missing!", _batch_idx)

        if batch_size and batch_size > 1:
            print(f"\n🔥 Parallel generation of {batch_size} images (Kie.ai Seedream 4.5)...\n")
            generated_images = []
            all_info         = []
            errors           = []
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {}
                for i in range(batch_size):
                    fut = executor.submit(
                        self._generate_seedream_kie,
                        prompt, image_tensors, image_size, aspect_ratio,
                        kie_api_key, ws_api_key, 1, disable_safety, i + 1,
                    )
                    futures[fut] = i
                for future in as_completed(futures):
                    idx = futures[future] + 1
                    try:
                        result = future.result()
                        generated_images.append(result[0])
                        all_info.append(result[1])
                    except Exception as e:
                        errors.append(f"Batch {idx}: {e}")
                        print(f"❌ [KSD Batch {idx}] Failed — {e}")

            if not generated_images:
                return self._handle_error(
                    f"All Seedream Kie generations failed: {'; '.join(errors)}"
                )
            ref_shape    = generated_images[0].shape
            valid_images = [img for img in generated_images if img.shape == ref_shape]
            if not valid_images:
                return self._handle_error("No valid image.")
            combined      = torch.cat(valid_images, dim=0)
            combined_info = "\n\n".join(all_info)
            if errors:
                combined_info = f"⚠️ {len(errors)} image(s) failed\n" + combined_info
            return (combined, combined_info, "")

        tag     = f"[KSD Batch {_batch_idx}]" if _batch_idx else "[Kie.ai Seedream]"
        headers = {
            "Authorization": f"Bearer {kie_api_key}",
            "Content-Type":  "application/json",
        }
        quality    = "high" if image_size == "4K" else "basic"
        image_urls = []
        if image_tensors:
            print(f"🖼️  {tag} Converting {len(image_tensors)} image(s)...")
            for i, tensor in enumerate(image_tensors, 1):
                url = self._tensor_to_public_url(tensor, idx=i, ws_api_key=ws_api_key, kie_api_key=kie_api_key)
                if url:
                    image_urls.append(url)
            # Fail-fast : si une seule ref a echoue a l'upload, on aborte plutot
            # que de laisser Kie generer en text-to-image silencieusement.
            if len(image_urls) < len(image_tensors):
                _missing = len(image_tensors) - len(image_urls)
                return self._handle_error(
                    f"❌ {tag} Upload failed: {len(image_urls)}/{len(image_tensors)} reference image(s) "
                    f"uploaded ({_missing} missing). Generation aborted to avoid producing an image without "
                    f"your references. Check network/firewall (catbox/litterbox/0x0.st/telegra.ph all failed).",
                    _batch_idx,
                )

        nsfw_checker = not disable_safety
        payload = {
            "model": KIE_SEEDREAM_MODEL,
            "input": {
                "prompt":       prompt,
                "image_urls":   image_urls,
                "aspect_ratio": aspect_ratio,
                "quality":      quality,
                "nsfw_checker": nsfw_checker,
            },
        }
        print(f"🚀 {tag} Submitting Seedream 4.5 (quality={quality})...")

        # Option B: session locale par batch (evite stale connections du pool global)
        # Option C: jitter 0-0.5s avant POST en batch (evite burst simultane)
        _local_kie = _make_kie_session()
        if _batch_idx is not None:
            import random as _random_jitter
            time.sleep(_random_jitter.uniform(0, 0.5))
        try:
            resp = _local_kie.post(KIE_CREATE_URL, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            return self._handle_error(f"❌ {tag} Submission error : {e}", _batch_idx)

        if data.get("code") != 200:
            return self._handle_error(
                f"❌ {tag} Submission failed: {data.get('msg', 'unknown error')}", _batch_idx
            )
        task_id = data.get("data", {}).get("taskId")
        if not task_id:
            return self._handle_error(f"❌ {tag} No taskId: {data}", _batch_idx)
        print(f"🔖 {tag} Task ID : {task_id}")

        elapsed     = 0
        result_data = {}
        print(f"⏳ {tag} Waiting for result (timeout: {KIE_TIMEOUT_S}s)...")
        while elapsed < KIE_TIMEOUT_S:
            time.sleep(KIE_POLL_DELAY)
            elapsed += KIE_POLL_DELAY
            try:
                poll_resp = _local_kie.get(
                    KIE_POLL_URL, params={"taskId": task_id},
                    headers=headers, timeout=15,
                )
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
            except requests.RequestException as e:
                print(f"⚠️  {tag} Polling error ({elapsed}s) : {e}")
                continue

            if poll_data.get("code") != 200:
                return self._handle_error(
                    f"❌ {tag} Poll error: {poll_data.get('msg')}", _batch_idx
                )
            result_data = poll_data.get("data", {})
            state       = result_data.get("state", "")
            if state == "success":
                print(f"✅ {tag} Task done! ({elapsed}s)")
                break
            elif state in ("fail", "failed", "error"):
                return self._handle_error(
                    f"❌ {tag} Failed: {result_data.get('failMsg', '?')}", _batch_idx
                )
            else:
                print(f"   {tag} [{elapsed}s/{KIE_TIMEOUT_S}s] state={state!r}...")
        else:
            return self._handle_error(f"❌ {tag} Timeout after {KIE_TIMEOUT_S}s.", _batch_idx)

        try:
            import json as _json
            result_json = _json.loads(result_data.get("resultJson", "{}"))
            result_urls = result_json.get("resultUrls", [])
            if not result_urls:
                return self._handle_error(f"❌ {tag} No URL in resultJson.", _batch_idx)
            output_url = result_urls[0]
        except Exception as e:
            return self._handle_error(f"❌ {tag} Parsing resultJson : {e}", _batch_idx)

        try:
            img_resp         = requests.get(output_url, timeout=60)
            img_resp.raise_for_status()
            pil_result       = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            image_np         = np.array(pil_result).astype(np.float32) / 255.0
            image_tensor_out = torch.from_numpy(image_np)[None,]
            cost_ms = result_data.get("costTime", 0)
            info = (
                f"[Kie.ai] Model: Seedream 4.5 Edit\n"
                f"Quality: {quality} | Ratio: {aspect_ratio}\n"
                f"Time: {cost_ms}ms | Total: {elapsed}s"
            )
            print(f"🎉 {tag} Done!\n{info}")
            return (image_tensor_out, info, "")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Download failed: {e}", _batch_idx)

    # ─────────────────────────────────────────────────────────────
    #  GPT IMAGE 2.0 — Kie.ai
    # ─────────────────────────────────────────────────────────────
    def _generate_gpt2_kie(
        self,
        prompt, image_tensors, aspect_ratio="1:1",
        image_size="2K",
        kie_api_key="", ws_api_key="",
        batch_size=1, disable_safety=False,
        _batch_idx=None,
    ):
        if not kie_api_key:
            return self._handle_error("❌ kie_api_key missing!", _batch_idx)

        if batch_size and batch_size > 1:
            print(f"\n🔥 Parallel generation of {batch_size} images (Kie.ai GPT Image 2.0)...\n")
            generated_images = []
            all_info         = []
            errors           = []
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {}
                for i in range(batch_size):
                    fut = executor.submit(
                        self._generate_gpt2_kie,
                        prompt, image_tensors, aspect_ratio,
                        image_size, kie_api_key, ws_api_key, 1, disable_safety, i + 1,
                    )
                    futures[fut] = i
                for future in as_completed(futures):
                    idx = futures[future] + 1
                    try:
                        result = future.result()
                        generated_images.append(result[0])
                        all_info.append(result[1])
                    except Exception as e:
                        errors.append(f"Batch {idx}: {e}")
                        print(f"❌ [KGP2 Batch {idx}] Failed — {e}")

            if not generated_images:
                return self._handle_error(
                    f"All GPT Image 2.0 Kie generations failed: {'; '.join(errors)}"
                )
            ref_shape    = generated_images[0].shape
            valid_images = [img for img in generated_images if img.shape == ref_shape]
            if not valid_images:
                return self._handle_error("No valid image.")
            combined      = torch.cat(valid_images, dim=0)
            combined_info = "\n\n".join(all_info)
            if errors:
                combined_info = f"⚠️ {len(errors)} image(s) failed\n" + combined_info
            return (combined, combined_info, "")

        tag     = f"[KGP2 Batch {_batch_idx}]" if _batch_idx else "[Kie.ai GPT Image 2.0]"
        headers = {
            "Authorization": f"Bearer {kie_api_key}",
            "Content-Type":  "application/json",
        }

        # Choisir le model selon présence ou non d'images source
        has_images = bool(image_tensors)
        kie_model  = "gpt-image-2-image-to-image" if has_images else "gpt-image-2-text-to-image"

        # Upload des images source via WaveSpeed si présentes
        image_urls = []
        if has_images:
            print(f"🖼️  {tag} Converting {len(image_tensors)} image(s)...")
            for i, tensor in enumerate(image_tensors, 1):
                url = self._tensor_to_public_url(tensor, idx=i, ws_api_key=ws_api_key, kie_api_key=kie_api_key)
                if url:
                    image_urls.append(url)
            # Fail-fast : aborte si toutes OU partiellement les refs ont echoue
            if not image_urls:
                return self._handle_error(
                    f"❌ {tag} Image upload failed (ws_api_key required for img2img).", _batch_idx
                )
            if len(image_urls) < len(image_tensors):
                _missing = len(image_tensors) - len(image_urls)
                return self._handle_error(
                    f"❌ {tag} Upload partiel: {len(image_urls)}/{len(image_tensors)} reference image(s) "
                    f"uploaded ({_missing} missing). Generation aborted to avoid producing an image without "
                    f"your full references.",
                    _batch_idx,
                )

        nsfw_checker = not disable_safety
        # KIE : 4K incompatible with 1:1 and auto aspect_ratio → fallback 2K
        _kie_resolution = image_size
        if image_size == "4K" and aspect_ratio in ("1:1", "auto"):
            print(f"⚠️  {tag} KIE GPT2: 4K incompatible with aspect_ratio={aspect_ratio!r} → fallback 2K")
            _kie_resolution = "2K"
        if has_images:
            payload = {
                "model": kie_model,
                "input": {
                    "prompt":       prompt,
                    "input_urls":   image_urls,
                    "aspect_ratio": aspect_ratio,
                    "resolution":   _kie_resolution,
                    "nsfw_checker": nsfw_checker,
                },
            }
        else:
            payload = {
                "model": kie_model,
                "input": {
                    "prompt":       prompt,
                    "aspect_ratio": aspect_ratio,
                    "resolution":   _kie_resolution,
                    "nsfw_checker": nsfw_checker,
                },
            }

        mode_label = "img2img" if has_images else "txt2img"
        print(f"🚀 {tag} Submitting GPT Image 2.0 [{mode_label}] (ratio={aspect_ratio})...")

        # Option B: session locale par batch (evite stale connections du pool global)
        # Option C: jitter 0-0.5s avant POST en batch (evite burst simultane)
        _local_kie = _make_kie_session()
        if _batch_idx is not None:
            import random as _random_jitter
            time.sleep(_random_jitter.uniform(0, 0.5))
        try:
            resp = _local_kie.post(KIE_CREATE_URL, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            return self._handle_error(f"❌ {tag} Submission error : {e}", _batch_idx)

        if data.get("code") != 200:
            return self._handle_error(
                f"❌ {tag} Submission failed: {data.get('msg', 'unknown error')}", _batch_idx
            )
        task_id = data.get("data", {}).get("taskId")
        if not task_id:
            return self._handle_error(f"❌ {tag} No taskId: {data}", _batch_idx)
        print(f"🔖 {tag} Task ID : {task_id}")

        elapsed     = 0
        result_data = {}
        print(f"⏳ {tag} Waiting for result (timeout: {KIE_TIMEOUT_S}s)...")
        while elapsed < KIE_TIMEOUT_S:
            time.sleep(KIE_POLL_DELAY)
            elapsed += KIE_POLL_DELAY
            try:
                poll_resp = _local_kie.get(
                    KIE_POLL_URL, params={"taskId": task_id},
                    headers=headers, timeout=15,
                )
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
            except requests.RequestException as e:
                print(f"⚠️  {tag} Polling error ({elapsed}s) : {e}")
                continue

            if poll_data.get("code") != 200:
                return self._handle_error(
                    f"❌ {tag} Poll error: {poll_data.get('msg')}", _batch_idx
                )
            result_data = poll_data.get("data", {})
            state       = result_data.get("state", "")
            if state == "success":
                print(f"✅ {tag} Task done! ({elapsed}s)")
                break
            elif state in ("fail", "failed", "error"):
                return self._handle_error(
                    f"❌ {tag} Failed: {result_data.get('failMsg', '?')}", _batch_idx
                )
            else:
                print(f"   {tag} [{elapsed}s/{KIE_TIMEOUT_S}s] state={state!r}...")
        else:
            return self._handle_error(f"❌ {tag} Timeout after {KIE_TIMEOUT_S}s.", _batch_idx)

        try:
            import json as _json
            result_json = _json.loads(result_data.get("resultJson", "{}"))
            result_urls = result_json.get("resultUrls", [])
            if not result_urls:
                return self._handle_error(f"❌ {tag} No URL in resultJson.", _batch_idx)
            output_url = result_urls[0]
        except Exception as e:
            return self._handle_error(f"❌ {tag} Parsing resultJson : {e}", _batch_idx)

        try:
            img_resp         = requests.get(output_url, timeout=60)
            img_resp.raise_for_status()
            pil_result       = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            image_np         = np.array(pil_result).astype(np.float32) / 255.0
            image_tensor_out = torch.from_numpy(image_np)[None,]
            cost_ms = result_data.get("costTime", 0)
            info = (
                f"[Kie.ai] Model: GPT Image 2.0 [{mode_label}]\n"
                f"Ratio: {aspect_ratio} | Time: {cost_ms}ms | Total: {elapsed}s"
            )
            print(f"🎉 {tag} Done!\n{info}")
            return (image_tensor_out, info, "")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Download failed: {e}", _batch_idx)

    # ─────────────────────────────────────────────────────────────
    #  SEEDREAM 4.5 — Fal.ai
    # ─────────────────────────────────────────────────────────────
    def _generate_seedream_fal(
        self,
        prompt, image_tensors, image_size, aspect_ratio,
        batch_size=1, fal_api_key="",
        disable_safety=False, _batch_idx=None,
    ):
        if not fal_api_key:
            return self._handle_error("❌ fal_api_key missing!", _batch_idx)
        if not _ensure_fal_client():
            return self._handle_error("❌ Unable to install fal-client.", _batch_idx)
        import fal_client
        os.environ["FAL_KEY"] = fal_api_key
        endpoint = FAL_SEEDREAM_ENDPOINT

        if batch_size and batch_size > 1:
            print(f"\n🔥 Parallel generation of {batch_size} images (Fal.ai Seedream 4.5)...\n")
            generated_images = []
            all_info         = []
            errors           = []
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {}
                for i in range(batch_size):
                    fut = executor.submit(
                        self._generate_seedream_fal,
                        prompt, image_tensors, image_size, aspect_ratio,
                        1, fal_api_key, disable_safety, i + 1,
                    )
                    futures[fut] = i
                for future in as_completed(futures):
                    idx = futures[future] + 1
                    try:
                        result = future.result()
                        generated_images.append(result[0])
                        all_info.append(result[1])
                    except Exception as e:
                        err_short = str(e)[:300] + "..." if len(str(e)) > 300 else str(e)
                        errors.append(f"Batch {idx}: {err_short}")
                        print(f"❌ [FSD Batch {idx}] Failed — content_policy_violation (prompt truncated from logs)")

            if not generated_images:
                return self._handle_error(
                    f"All Seedream FAL generations failed: {'; '.join(errors)}"
                )
            ref_shape    = generated_images[0].shape
            valid_images = [img for img in generated_images if img.shape == ref_shape]
            if not valid_images:
                return self._handle_error("No valid image.")
            combined      = torch.cat(valid_images, dim=0)
            combined_info = "\n\n".join(all_info)
            if errors:
                combined_info = f"⚠️ {len(errors)} image(s) failed\n" + combined_info
            return (combined, combined_info, "")

        tag            = f"[FSD Batch {_batch_idx}]" if _batch_idx else "[Fal.ai Seedream]"
        fal_image_size = "auto_4K" if image_size == "4K" else "auto_2K"

        print(f"⬆️  {tag} Uploading {len(image_tensors)} image(s)...")
        image_urls = []
        for i, tensor in enumerate(image_tensors, 1):
            try:
                url = _tensor_to_fal_url(tensor, fal_client, tag=tag, idx=i)
                image_urls.append(url)
            except Exception as e:
                return self._handle_error(f"❌ {tag} Image upload error {i} : {e}", _batch_idx)

        arguments = {
            "prompt":                prompt,
            "image_urls":            image_urls,
            "image_size":            fal_image_size,
            "num_images":            1,
            "enable_safety_checker": not disable_safety,
        }
        print(f"🚀 {tag} Submitting → {endpoint}")

        try:
            handler    = fal_client.submit(endpoint, arguments=arguments)
            request_id = handler.request_id
            elapsed    = 0
            while elapsed < FAL_TIMEOUT_S:
                time.sleep(FAL_POLL_DELAY)
                elapsed += FAL_POLL_DELAY
                try:
                    status = fal_client.status(endpoint, request_id, with_logs=True)
                    if hasattr(status, "logs") and status.logs:
                        for log in status.logs:
                            msg = log.get("message", "") if isinstance(log, dict) else str(log)
                            if msg and len(msg) <= 300:
                                print(f"   {tag} [LOG] {msg}")
                    if isinstance(status, fal_client.Completed):
                        print(f"✅ {tag} Done! ({elapsed}s)")
                        break
                    elif isinstance(status, fal_client.Queued):
                        pos = getattr(status, "position", "?")
                        print(f"   {tag} [{elapsed}s] Queue position={pos}")
                    else:
                        print(f"   {tag} [{elapsed}s] En cours...")
                except Exception as e:
                    print(f"⚠️  {tag} Status error : {e}")
            else:
                return self._handle_error(f"❌ {tag} Timeout after {FAL_TIMEOUT_S}s.", _batch_idx)

            result_data = fal_client.result(endpoint, request_id)
            fal_images  = result_data.get("images", [])
            if not fal_images:
                return self._handle_error(f"❌ {tag} Aucune image : {result_data}", _batch_idx)
            img_info   = fal_images[0]
            output_url = img_info.get("url", "")
            if not output_url:
                return self._handle_error(f"❌ {tag} URL missing.", _batch_idx)

            img_resp         = requests.get(output_url, timeout=60)
            img_resp.raise_for_status()
            pil_result       = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            image_np         = np.array(pil_result).astype(np.float32) / 255.0
            image_tensor_out = torch.from_numpy(image_np)[None,]
            info = (
                f"[Fal.ai] Model: Seedream 4.5 Edit\n"
                f"Taille : {fal_image_size} | {img_info.get('width','?')}x{img_info.get('height','?')}px\n"
                f"Temps total : {elapsed}s"
            )
            print(f"🎉 {tag} Done!\n{info}")
            return (image_tensor_out, info, "")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Error: {type(e).__name__} : {e}", _batch_idx)

    # ─────────────────────────────────────────────────────────────
    #  GPT IMAGE 2.0 — Fal.ai
    # ─────────────────────────────────────────────────────────────
    def _generate_gpt2_fal(
        self,
        prompt, image_tensors, aspect_ratio="1:1", image_size="2K",
        gpt2_quality="high", batch_size=1, fal_api_key="",
        _batch_idx=None,
    ):
        if not fal_api_key:
            return self._handle_error("❌ fal_api_key missing!", _batch_idx)
        if not _ensure_fal_client():
            return self._handle_error("❌ Unable to install fal-client.", _batch_idx)
        import fal_client
        os.environ["FAL_KEY"] = fal_api_key

        if batch_size and batch_size > 1:
            print(f"\n🔥 Parallel generation of {batch_size} images (Fal.ai GPT Image 2.0)...\n")
            generated_images = []
            all_info         = []
            errors           = []
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {}
                for i in range(batch_size):
                    fut = executor.submit(
                        self._generate_gpt2_fal,
                        prompt, image_tensors, aspect_ratio, image_size,
                        gpt2_quality, 1, fal_api_key, i + 1,
                    )
                    futures[fut] = i
                for future in as_completed(futures):
                    idx = futures[future] + 1
                    try:
                        result = future.result()
                        generated_images.append(result[0])
                        all_info.append(result[1])
                    except Exception as e:
                        short_e = _fal_short_error(e)
                        errors.append(f"Batch {idx}: {short_e}")
                        print(f"❌ [FGP2 Batch {idx}] Failed — {short_e}")

            if not generated_images:
                return self._handle_error(
                    f"All GPT Image 2.0 FAL generations failed:\n" + "\n".join(errors)
                )
            ref_shape    = generated_images[0].shape
            valid_images = [img for img in generated_images if img.shape == ref_shape]
            if not valid_images:
                return self._handle_error("No valid image.")
            combined      = torch.cat(valid_images, dim=0)
            combined_info = "\n\n".join(all_info)
            if errors:
                combined_info = f"⚠️ {len(errors)} image(s) failed\n" + combined_info
            return (combined, combined_info, "")

        has_images = bool(image_tensors)
        endpoint   = FAL_GPT2_EDIT_ENDPOINT if has_images else FAL_GPT2_TXT2IMG_ENDPOINT
        tag        = f"[FGP2 Batch {_batch_idx}]" if _batch_idx else "[Fal.ai GPT Image 2.0]"
        mode_label = "img2img" if has_images else "txt2img"

        # Résolution : lookup dans la table, fallback "auto" pour edit
        fal_size = _GPT2_FAL_SIZE_MAP.get((aspect_ratio, image_size))
        if fal_size is None:
            fal_size = "auto" if has_images else "landscape_4_3"
            print(f"⚠️  {tag} Combo ({aspect_ratio}, {image_size}) not mapped → fallback {fal_size!r}")

        # Upload images si présentes
        image_urls = []
        if has_images:
            print(f"⬆️  {tag} Uploading {len(image_tensors)} image(s)...")
            for i, tensor in enumerate(image_tensors, 1):
                try:
                    url = _tensor_to_fal_url(tensor, fal_client, tag=tag, idx=i)
                    image_urls.append(url)
                except Exception as e:
                    return self._handle_error(f"❌ {tag} Image upload error {i} : {e}", _batch_idx)

        arguments = {
            "prompt":       prompt,
            "image_size":   fal_size,
            "quality":      gpt2_quality,
            "num_images":   1,
            "output_format": "png",
        }
        if has_images:
            arguments["image_urls"] = image_urls

        print(f"🚀 {tag} Submitting → {endpoint} [{mode_label}] (size={fal_size!r}, quality={gpt2_quality})")

        try:
            handler    = fal_client.submit(endpoint, arguments=arguments)
            request_id = handler.request_id
            elapsed    = 0
            while elapsed < FAL_TIMEOUT_S:
                time.sleep(FAL_POLL_DELAY)
                elapsed += FAL_POLL_DELAY
                try:
                    status = fal_client.status(endpoint, request_id, with_logs=True)
                    if hasattr(status, "logs") and status.logs:
                        for log in status.logs:
                            msg = log.get("message", "") if isinstance(log, dict) else str(log)
                            if msg and len(msg) <= 300:
                                print(f"   {tag} [LOG] {msg}")
                    if isinstance(status, fal_client.Completed):
                        print(f"✅ {tag} Done! ({elapsed}s)")
                        break
                    elif isinstance(status, fal_client.Queued):
                        pos = getattr(status, "position", "?")
                        print(f"   {tag} [{elapsed}s] Queue position={pos}")
                    else:
                        print(f"   {tag} [{elapsed}s] En cours...")
                except Exception as e:
                    print(f"⚠️  {tag} Status error : {e}")
            else:
                return self._handle_error(f"❌ {tag} Timeout after {FAL_TIMEOUT_S}s.", _batch_idx)

            result_data = fal_client.result(endpoint, request_id)
            fal_images  = result_data.get("images", [])
            if not fal_images:
                return self._handle_error(f"❌ {tag} Aucune image : {result_data}", _batch_idx)
            img_info   = fal_images[0]
            output_url = img_info.get("url", "")
            if not output_url:
                return self._handle_error(f"❌ {tag} URL missing.", _batch_idx)

            img_resp         = requests.get(output_url, timeout=60)
            img_resp.raise_for_status()
            pil_result       = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            image_np         = np.array(pil_result).astype(np.float32) / 255.0
            image_tensor_out = torch.from_numpy(image_np)[None,]
            info = (
                f"[Fal.ai] Model: GPT Image 2.0 [{mode_label}]\n"
                f"Taille : {fal_size} | {img_info.get('width','?')}x{img_info.get('height','?')}px\n"
                f"Quality: {gpt2_quality} | Total time: {elapsed}s"
            )
            print(f"🎉 {tag} Done!\n{info}")
            return (image_tensor_out, info, "")
        except Exception as e:
            return self._handle_error(
                f"❌ {tag} Error: {type(e).__name__} : {_fal_short_error(e)}", _batch_idx
            )

    # ─────────────────────────────────────────────────────────────
    #  FAL.AI — NB Pro / NB2
    # ─────────────────────────────────────────────────────────────
    def _generate_fal(
        self,
        prompt, image_tensors, image_size, aspect_ratio,
        batch_size=1, fal_api_key="", is_nb2=False,
        safety_tolerance="4", enable_web_search=False,
        negative_prompt="",
        _batch_idx=None,
    ):
        if not fal_api_key:
            return self._handle_error("❌ fal_api_key missing!", _batch_idx)
        if not _ensure_fal_client():
            return self._handle_error("❌ Unable to install fal-client.", _batch_idx)
        import fal_client
        os.environ["FAL_KEY"] = fal_api_key

        has_images = bool(image_tensors)
        if is_nb2:
            endpoint = FAL_NB2_EDIT_ENDPOINT if has_images else FAL_NB2_T2I_ENDPOINT
        else:
            endpoint = FAL_PRO_EDIT_ENDPOINT if has_images else FAL_PRO_T2I_ENDPOINT

        model_label = endpoint
        mode_label  = "img2img" if has_images else "text-to-image"
        print(f"ℹ️  [Fal.ai] Mode : {mode_label} → {endpoint}")

        if batch_size and batch_size > 1:
            print(f"\n🔥 Parallel generation of {batch_size} images (Fal.ai)...\n")
            generated_images = []
            all_info         = []
            errors           = []
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {}
                for i in range(batch_size):
                    fut = executor.submit(
                        self._generate_fal,
                        prompt, image_tensors, image_size, aspect_ratio,
                        1, fal_api_key, is_nb2, safety_tolerance, enable_web_search,
                        negative_prompt, i + 1,
                    )
                    futures[fut] = i
                for future in as_completed(futures):
                    idx = futures[future] + 1
                    try:
                        result = future.result()
                        generated_images.append(result[0])
                        all_info.append(result[1])
                    except Exception as e:
                        errors.append(f"Batch {idx}: {e}")
                        print(f"❌ [FAL Batch {idx}] Failed — {e}")

            if not generated_images:
                return self._handle_error(
                    f"All Fal.ai generations failed: {'; '.join(errors)}"
                )
            ref_shape    = generated_images[0].shape
            valid_images = [img for img in generated_images if img.shape == ref_shape]
            if not valid_images:
                return self._handle_error("No valid image.")
            combined      = torch.cat(valid_images, dim=0)
            combined_info = "\n\n".join(all_info)
            if errors:
                combined_info = f"⚠️ {len(errors)} image(s) failed\n" + combined_info
            return (combined, combined_info, "")

        tag = f"[FAL Batch {_batch_idx}]" if _batch_idx else "[Fal.ai]"
        arguments = {
            "prompt":            prompt,
            "resolution":        image_size,
            "aspect_ratio":      aspect_ratio,
            "output_format":     "png",
            "safety_tolerance":  safety_tolerance,
            "enable_web_search": enable_web_search,
            "limit_generations": True,
            "num_images":        1,
        }
        if negative_prompt and negative_prompt.strip():
            arguments["negative_prompt"] = negative_prompt.strip()

        if has_images:
            print(f"⬆️  {tag} Uploading {len(image_tensors)} image(s)...")
            image_urls = []
            for i, tensor in enumerate(image_tensors, 1):
                try:
                    url = _tensor_to_fal_url(tensor, fal_client, tag=tag, idx=i)
                    image_urls.append(url)
                except Exception as e:
                    return self._handle_error(f"❌ {tag} Image upload error {i} : {e}", _batch_idx)
            arguments["image_urls"] = image_urls

        print(f"🚀 {tag} Submitting → {endpoint} [{mode_label}]")

        import time as _time
        _t0 = _time.time()
        _last_log_t = [0.0]

        def _on_update(update):
            elapsed_now = int(_time.time() - _t0)
            now = _time.time()
            if isinstance(update, fal_client.InProgress):
                for log in (update.logs or []):
                    msg = log.get("message", "") if isinstance(log, dict) else str(log)
                    if msg and len(msg) <= 300 and now - _last_log_t[0] >= 10:
                        print(f"   {tag} [LOG] {msg}")
                        _last_log_t[0] = now
            elif isinstance(update, fal_client.Queued):
                if now - _last_log_t[0] >= 10:
                    pos = getattr(update, "position", "?")
                    print(f"   {tag} [{elapsed_now}s] Queue position={pos}")
                    _last_log_t[0] = now
            else:
                if now - _last_log_t[0] >= 10:
                    print(f"   {tag} [{elapsed_now}s] En cours...")
                    _last_log_t[0] = now

        try:
            result_data = fal_client.subscribe(
                endpoint,
                arguments=arguments,
                with_logs=True,
                on_queue_update=_on_update,
            )
            elapsed = int(_time.time() - _t0)

            fal_images = result_data.get("images", [])
            if not fal_images:
                return self._handle_error(f"❌ {tag} Aucune image : {result_data}", _batch_idx)
            img_info   = fal_images[0]
            output_url = img_info.get("url", "")
            if not output_url:
                return self._handle_error(f"❌ {tag} URL missing.", _batch_idx)

            print(f"✅ {tag} Image ready! ({elapsed}s)")
            img_resp         = requests.get(output_url, timeout=60)
            img_resp.raise_for_status()
            pil_result       = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            image_np         = np.array(pil_result).astype(np.float32) / 255.0
            image_tensor_out = torch.from_numpy(image_np)[None,]
            w = img_info.get("width", "?")
            h = img_info.get("height", "?")
            info = (
                f"[Fal.ai] {model_label} [{mode_label}]\n"
                f"Taille : {image_size} | {w}x{h}px | Temps total : {elapsed}s"
            )
            print(f"🎉 {tag} Done!\n{info}")
            return (image_tensor_out, info, "")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Error: {type(e).__name__} : {e}", _batch_idx)

    # ─────────────────────────────────────────────────────────────
    #  NANO BANANA 2 — Google REST
    # ─────────────────────────────────────────────────────────────
    def _generate_nb2_google(
        self,
        prompt, image_tensors, image_size, aspect_ratio,
        temperature=1.0, g_key="", batch_size=1,
        disable_safety=False, _batch_idx=None,
    ):
        if batch_size and batch_size > 1:
            print(f"\n🔥 Parallel generation of {batch_size} images (Google NB2)...\n")
            generated_images   = []
            all_text_responses = []
            errors             = []
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {
                    executor.submit(
                        self._generate_nb2_google,
                        prompt, image_tensors, image_size, aspect_ratio,
                        temperature, g_key, 1, disable_safety, i + 1,
                    ): i for i in range(batch_size)
                }
                _FTE = __import__('concurrent.futures', fromlist=['TimeoutError']).TimeoutError
                _proc = set()
                try:
                    for future in as_completed(futures, timeout=NB2_TIMEOUT_S + 30):
                        _proc.add(future)
                        idx = futures[future] + 1
                        try:
                            result = future.result(timeout=10)
                            generated_images.append(result[0])
                            all_text_responses.append(result[1])
                        except Exception as e:
                            errors.append(f"Batch {idx}: {e}")
                            print(f"❌ [NB2 Batch {idx}] Failed — {e}")
                except _FTE:
                    print("[Google NB2] timeout — salvaging completed futures...")
                    for _f, _i in futures.items():
                        if _f in _proc: continue
                        _fi = _i + 1
                        if _f.done():
                            try:
                                _r = _f.result(timeout=0)
                                generated_images.append(_r[0])
                                all_text_responses.append(_r[1])
                                print(f"✅ [NB2 Batch {_fi}] Recovered")
                            except Exception as _fe:
                                errors.append(f"Batch {_fi}: {_fe}")
                        else:
                            errors.append(f"Batch {_fi}: outer timeout")

            if not generated_images:
                return self._handle_error(
                    f"All NB2 generations failed: {'; '.join(errors)}"
                )
            ref_shape    = generated_images[0].shape
            valid_images = [img for img in generated_images if img.shape == ref_shape]
            if not valid_images:
                return self._handle_error("No valid image.")
            combined      = torch.cat(valid_images, dim=0)
            combined_text = "\n\n".join(all_text_responses)
            if errors:
                combined_text = f"⚠️ {len(errors)} error(s)\n" + combined_text
            return (combined, combined_text, "")

        tag   = f"[NB2 Batch {_batch_idx}]" if _batch_idx else "[Google NB2]"
        parts = []
        for i, img_tensor in enumerate(image_tensors, 1):
            print(f"📷 {tag} Adding image {i}...")
            parts.append(_tensor_to_base64_part(img_tensor))
        parts.append({"text": prompt})

        image_config = {}
        if aspect_ratio and aspect_ratio != "auto":
            image_config["aspectRatio"] = aspect_ratio
        if image_size:
            image_config["imageSize"] = image_size

        gen_config: dict = {"responseModalities": ["TEXT", "IMAGE"], "temperature": temperature}
        if image_config:
            gen_config["imageConfig"] = image_config

        safety_settings = [
            {"category": cat, "threshold": "OFF"} for cat in _HARM_CATEGORIES
        ]
        payload = {
            "contents":          [{"role": "user", "parts": parts}],
            "systemInstruction": {"parts": [{"text": NB2_SYSTEM_PROMPT}]},
            "generationConfig":  gen_config,
            "safetySettings":    safety_settings,
        }
        url = f"{NB2_BASE_URL}?key={g_key}"

        print(f"⏳ {tag} Sending NB2 request (timeout: {NB2_TIMEOUT_S}s)...")
        try:
            with _Heartbeat(f"{tag} In progress"):
                response = requests.post(
                    url, json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=NB2_TIMEOUT_S,
                )
                response.raise_for_status()
                try:
                    data = response.json()
                except ValueError as e:
                    return self._handle_error(f"❌ {tag} Invalid JSON response : {e}", _batch_idx)

            image_bytes = _extract_image_bytes_from_nb2_response(data)

            if not image_bytes:
                candidates   = data.get("candidates", [])
                if candidates:
                    finish = candidates[0].get("finishReason", "?")
                    safety = candidates[0].get("safetyRatings", [])
                    blocked = [r for r in safety if r.get("blocked")]
                    detail = f", blocked by: {blocked}" if blocked else ""
                    return self._handle_error(
                        f"❌ {tag} Aucune image (finishReason={finish}{detail}).", _batch_idx
                    )
                else:
                    feedback     = data.get("promptFeedback", {})
                    block_reason = feedback.get("blockReason", "?")
                    return self._handle_error(
                        f"❌ {tag} Request blocked by API (blockReason={block_reason}).", _batch_idx
                    )

            pil_image    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_np     = np.array(pil_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            text_response = ""
            for cand in data.get("candidates", []):
                for part in cand.get("content", {}).get("parts", []):
                    if "text" in part:
                        text_response += part["text"]

            print(f"✅ {tag} Generation successful!")
            return (image_tensor, text_response, "")

        except requests.exceptions.Timeout:
            return self._handle_error(f"❌ {tag} Timeout after {NB2_TIMEOUT_S}s.", _batch_idx)
        except requests.exceptions.HTTPError as e:
            try:
                body = e.response.json()
            except Exception:
                body = e.response.text
            return self._handle_error(f"❌ {tag} HTTP {e.response.status_code} : {body}", _batch_idx)
        except requests.exceptions.RequestException as e:
            return self._handle_error(f"❌ {tag} Network error : {e}", _batch_idx)

    # ─────────────────────────────────────────────────────────────
    #  Helpers Google SDK
    # ─────────────────────────────────────────────────────────────
    def _detect_approach(self, g_key: str, v_proj: str, v_loc: str) -> str:
        if g_key:
            return "API"
        raise RuntimeError(
            "❌ No Google API key configured!\n"
            "→ Connect an 'API Keys Loader' node and wire gemini_api_key."
        )

    def _make_google_client(self, approach, g_key, v_proj="", v_loc="", model_name=""):
        if not g_key:
            raise RuntimeError("❌ gemini_api_key missing !")
        # Note: HttpOptions(timeout=...) retiré — certaines versions du SDK
        # google-genai propagent ce timeout comme kwarg `timeout=` sur
        # generate_content(), ce qui plante en mode batch parallèle.
        # Le timeout est déjà géré par _Heartbeat(max_duration=...) côté batch.
        return genai.Client(api_key=g_key)

    def _create_config(
        self,
        aspect_ratio, image_size, temperature, top_p,
        use_search, model_name,
        system_instructions=None, safety_threshold=None,
    ):
        if "preview" in model_name and not self._preview_warning_shown:
            print(f"⚠️  Using a preview model: {model_name}")
            self._preview_warning_shown = True

        if safety_threshold is not None:
            safety_settings = [
                types.SafetySetting(category=cat, threshold=safety_threshold)
                for cat in _HARM_CATEGORIES
            ]
        else:
            safety_settings = None

        config_kwargs = dict(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio, image_size=image_size),
            temperature=temperature,
            top_p=top_p,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        # Disable thinking mode — gemini-3-pro-image-preview uses its full token
        # budget "reasoning" and never outputs an image if thinking is enabled.
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            print("🧠 [Config] ThinkingConfig(thinking_budget=0) applied via class")
        except AttributeError:
            # Older SDK: pass as raw dict — google-genai accepts dicts for nested configs
            try:
                config_kwargs["thinking_config"] = {"thinking_budget": 0}
                print("🧠 [Config] ThinkingConfig applied via dict fallback")
            except Exception as _te:
                print(f"⚠️  [Config] Could not disable thinking mode: {_te} — consider: pip install --upgrade google-genai")
        if safety_settings is not None:
            config_kwargs["safety_settings"] = safety_settings

        config = types.GenerateContentConfig(**config_kwargs)

        if system_instructions and system_instructions.strip():
            config.system_instruction = system_instructions

        if use_search:
            try:
                config.tools = [types.Tool(google_search=types.GoogleSearch())]
            except Exception as e:
                print(f"⚠️  Search tool not supported: {e}")

        return config

    # ─────────────────────────────────────────────────────────────
    #  GOOGLE SDK — génération simple
    # ─────────────────────────────────────────────────────────────
    def _generate_single_image(
        self,
        model_name, prompt, use_search, approach, contents,
        aspect_ratio, image_size, temperature, top_p,
        system_instructions=None,
        g_key="", v_proj="", v_loc="", safety_threshold=None,
    ):
        client   = self._make_google_client(approach, g_key, v_proj, v_loc, model_name)
        config   = self._create_config(
            aspect_ratio, image_size, temperature, top_p,
            use_search, model_name, system_instructions, safety_threshold,
        )
        print("🚀 Google — generation in progress...")
        with _Heartbeat("[Google] In progress"):
            response = client.models.generate_content(
                model=model_name, contents=contents, config=config
            )
        if not response.candidates:
            reason = "?"
            try:
                pf = response.prompt_feedback
                if pf:
                    reason = str(getattr(pf, "block_reason", None) or getattr(pf, "blockReason", "?"))
            except Exception:
                pass
            return self._handle_error(f"Google API returned no candidates (blockReason={reason}).")

        cand = response.candidates[0]
        _VALID_FINISH = {
            types.FinishReason.STOP,
            getattr(types.FinishReason, "IMAGE_GENERATION_STOP", types.FinishReason.STOP),
        }
        if hasattr(cand, "finish_reason") and cand.finish_reason not in _VALID_FINISH:
            return self._handle_error(f"Google generation failed : {cand.finish_reason}")

        image_bytes   = None
        text_response = ""
        for part in cand.content.parts:
            if part.inline_data and image_bytes is None:
                image_bytes = part.inline_data.data
            elif part.text:
                text_response += part.text

        grounding_sources = self.extract_grounding_data(response)

        if image_bytes is None:
            return self._handle_error("No image data in Google response.")

        pil_image    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np     = np.array(pil_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        print("✅ Google — generation successful!")
        return (image_tensor, text_response, grounding_sources)

    # ─────────────────────────────────────────────────────────────
    #  GOOGLE SDK — tâche unitaire batch
    # ─────────────────────────────────────────────────────────────
    def _generate_one_image_task(
        self, task_id, model_name, contents, config,
        approach, g_key="", v_proj="", v_loc="",
    ):
        try:
            print(f"🚀 [Batch {task_id}] Starting...")
            client   = self._make_google_client(approach, g_key, v_proj, v_loc, model_name)
            with _Heartbeat(f"[Batch {task_id}] In progress",
                            max_duration=GOOGLE_TASK_TIMEOUT_S):
                # NB: le timeout est appliqué via HttpOptions à la création du
                # client (cf. _make_google_client). La SDK google-genai
                # n'accepte PAS de kwarg `timeout` sur generate_content().
                response = client.models.generate_content(
                    model=model_name, contents=contents, config=config,
                )
            if not response.candidates:
                reason = "?"
                try:
                    pf = response.prompt_feedback
                    if pf:
                        reason = str(getattr(pf, "block_reason", None) or getattr(pf, "blockReason", "?"))
                except Exception:
                    pass
                err = f"No candidates — request blocked (blockReason={reason})"
                print(f"⚠️  [Batch {task_id}] {err}")
                return {"success": False, "error": err,
                        "image_tensor": None, "text_response": "", "grounding_sources": ""}
            cand = response.candidates[0]
            _VALID_FINISH = {
                types.FinishReason.STOP,
                getattr(types.FinishReason, "IMAGE_GENERATION_STOP", types.FinishReason.STOP),
            }
            if hasattr(cand, "finish_reason") and cand.finish_reason not in _VALID_FINISH:
                err = f"finish_reason inattendu : {cand.finish_reason}"
                print(f"⚠️  [Batch {task_id}] {err}")
                return {"success": False, "error": err,
                        "image_tensor": None, "text_response": "", "grounding_sources": ""}

            image_bytes   = None
            text_response = ""
            for part in cand.content.parts:
                if part.inline_data and image_bytes is None:
                    image_bytes = part.inline_data.data
                elif part.text:
                    text_response += part.text

            grounding_sources = self.extract_grounding_data(response)

            if image_bytes is None:
                err = f"No image data (finish_reason={getattr(cand, 'finish_reason', '?')})"
                print(f"⚠️  [Batch {task_id}] {err}")
                return {"success": False, "error": err, "image_tensor": None,
                        "text_response": text_response, "grounding_sources": grounding_sources}

            pil_image    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_np     = np.array(pil_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            print(f"✅ [Batch {task_id}] Success!")
            return {"success": True, "image_tensor": image_tensor,
                    "text_response": text_response, "grounding_sources": grounding_sources, "error": None}
        except Exception as e:
            print(f"❌ [Batch {task_id}] Exception : {e}")
            return {"success": False, "error": str(e), "image_tensor": None,
                    "text_response": "", "grounding_sources": ""}

    # ─────────────────────────────────────────────────────────────
    #  GOOGLE SDK — batch parallèle
    # ─────────────────────────────────────────────────────────────
    def _generate_batch_parallel(
        self,
        model_name, prompt, batch_size, use_search, approach, contents,
        aspect_ratio, image_size, temperature, top_p,
        system_instructions=None,
        g_key="", v_proj="", v_loc="", safety_threshold=None,
    ):
        print(f"\n🔥 Parallel generation of {batch_size} images (Google)...\n")
        config = self._create_config(
            aspect_ratio, image_size, temperature, top_p,
            use_search, model_name, system_instructions, safety_threshold,
        )
        generated_images      = []
        all_text_responses    = []
        all_grounding_sources = []
        errors                = []

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {
                executor.submit(
                    self._generate_one_image_task,
                    i + 1, model_name, contents, copy.deepcopy(config), approach, g_key, v_proj, v_loc,
                ): i for i in range(batch_size)
            }
            try:
                for future in as_completed(futures, timeout=GOOGLE_TASK_TIMEOUT_S):
                    task_id = futures[future] + 1
                    try:
                        result = future.result(timeout=10)
                    except Exception as e:
                        err_msg = str(e)
                        errors.append(f"Batch {task_id}: {err_msg}")
                        print(f"❌ [Batch {task_id}] Failed — {err_msg}")
                        continue
                    if result["success"]:
                        generated_images.append(result["image_tensor"])
                        all_text_responses.append(result["text_response"])
                        all_grounding_sources.append(result["grounding_sources"])
                    else:
                        err_msg = result['error']
                        errors.append(f"Batch {task_id}: {err_msg}")
                        print(f"❌ [Batch {task_id}] Failed — {err_msg}")
            except TimeoutError:
                # Une ou plusieurs tâches n'ont pas répondu dans le délai imparti
                timed_out = [futures[f] + 1 for f in futures if not f.done()]
                for tid in timed_out:
                    msg = f"Timeout after {GOOGLE_TASK_TIMEOUT_S}s (Google SDK did not respond)"
                    errors.append(f"Batch {tid}: {msg}")
                    print(f"❌ [Batch {tid}] {msg}")

        print(f"\n✨ Done! {len(generated_images)}/{batch_size} successful\n")
        if errors:
            print(f"⚠️  Failures ({len(errors)}): {' | '.join(errors)}\n")
        if not generated_images:
            return self._handle_error(f"All generations failed : {'; '.join(errors)}")
        ref_shape    = generated_images[0].shape
        valid_images = [img for img in generated_images if img.shape == ref_shape]
        if not valid_images:
            return self._handle_error("No valid image.")
        combined_images    = torch.cat(valid_images, dim=0)
        combined_text      = "\n\n".join(all_text_responses)
        combined_grounding = "\n\n".join(all_grounding_sources)
        return (combined_images, combined_text, combined_grounding)

    # ─────────────────────────────────────────────────────────────
    #  WAVESPEED — NB Pro / NB2
    # ─────────────────────────────────────────────────────────────
    def _generate_wavespeed(
        self,
        prompt, image_tensors, image_size, output_format,
        aspect_ratio="1:1", batch_size=1,
        ws_api_key="", is_nb2=False,
        negative_prompt="",
        _batch_idx=None,
    ):
        if not ws_api_key:
            return self._handle_error("❌ wavespeed_api_key missing!", _batch_idx)
        if not image_tensors:
            return self._handle_error("❌ WaveSpeed requires at least one source image.", _batch_idx)

        # NB Pro 4K/8K → edit-ultra endpoint
        _use_ultra = (not is_nb2) and (image_size in ("4K", "8K"))
        if is_nb2:
            submit_url  = WAVESPEED_SUBMIT_NB2_URL
            model_label = "nano-banana-2/edit"
        elif _use_ultra:
            submit_url  = WAVESPEED_SUBMIT_PRO_ULTRA_URL
            model_label = "nano-banana-pro/edit-ultra"
        else:
            submit_url  = WAVESPEED_SUBMIT_PRO_URL
            model_label = "nano-banana-pro/edit"

        if batch_size and batch_size > 1:
            print(f"\n🔥 Parallel generation of {batch_size} images (WaveSpeed)...\n")
            generated_images = []
            all_info         = []
            errors           = []
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {}
                for i in range(batch_size):
                    fut = executor.submit(
                        self._generate_wavespeed,
                        prompt, image_tensors, image_size, output_format,
                        aspect_ratio, 1, ws_api_key, is_nb2, negative_prompt, i + 1,
                    )
                    futures[fut] = i
                for future in as_completed(futures):
                    idx = futures[future] + 1
                    try:
                        result = future.result()
                        generated_images.append(result[0])
                        all_info.append(result[1])
                    except Exception as e:
                        errors.append(f"Batch {idx}: {e}")
                        print(f"❌ [WS Batch {idx}] Failed — {e}")

            if not generated_images:
                return self._handle_error(
                    f"All WaveSpeed generations failed: {'; '.join(errors)}"
                )
            ref_shape    = generated_images[0].shape
            valid_images = [img for img in generated_images if img.shape == ref_shape]
            if not valid_images:
                return self._handle_error("No valid image.")
            combined      = torch.cat(valid_images, dim=0)
            combined_info = "\n\n".join(all_info)
            if errors:
                combined_info = f"⚠️ {len(errors)} image(s) failed\n" + combined_info
            return (combined, combined_info, "")

        tag          = f"[WS Batch {_batch_idx}]" if _batch_idx else "[WaveSpeed]"
        resolution   = SIZE_TO_RESOLUTION.get(image_size, "2k")
        headers_auth = {"Authorization": f"Bearer {ws_api_key}"}

        print(f"⬆️  {tag} Uploading {len(image_tensors)} image(s)...")
        image_urls = []
        for idx, img_tensor in enumerate(image_tensors, 1):
            try:
                pil_image  = tensor_to_pil(img_tensor)
                img_buffer = io.BytesIO()
                pil_image.save(img_buffer, format="PNG")
                img_buffer.seek(0)
                upload_resp = requests.post(
                    WAVESPEED_UPLOAD_URL, headers=headers_auth,
                    files={"file": (f"source_{idx}.png", img_buffer, "image/png")},
                    timeout=60,
                )
                upload_resp.raise_for_status()
                upload_data = upload_resp.json()
                image_url   = upload_data.get("data", {}).get("download_url")
                if not image_url:
                    return self._handle_error(
                        f"❌ {tag} Image {idx} upload failed : {upload_data}", _batch_idx
                    )
                image_urls.append(image_url)
                print(f"✅ {tag} Image {idx}/{len(image_tensors)} uploaded")
            except requests.RequestException as e:
                return self._handle_error(f"❌ {tag} Image upload error {idx} : {e}", _batch_idx)

        payload = {
            "prompt":               prompt,
            "images":               image_urls,
            "aspect_ratio":         aspect_ratio,
            "resolution":           resolution,
            "output_format":        output_format,
            "enable_base64_output": False,
            "enable_sync_mode":     False,
            "enable_web_search":    False,
            "enable_image_search":  False,
        }
        if negative_prompt and negative_prompt.strip():
            payload["negative_prompt"] = negative_prompt.strip()
        print(f"🚀 {tag} Submitting ({model_label}, resolution={resolution})...")

        try:
            submit_resp = requests.post(
                submit_url, json=payload,
                headers={**headers_auth, "Content-Type": "application/json"},
                timeout=30,
            )
            submit_resp.raise_for_status()
            submit_data = submit_resp.json()
            task_id = (submit_data.get("data") or {}).get("id") or submit_data.get("id")
            if not task_id:
                return self._handle_error(f"❌ {tag} Submission failed: {submit_data}", _batch_idx)
            print(f"🔖 {tag} Task ID : {task_id}")
        except requests.RequestException as e:
            body = ""
            try:
                body = e.response.text
            except Exception:
                pass
            return self._handle_error(f"❌ {tag} Submission error : {e} | Body: {body}", _batch_idx)

        poll_url  = WAVESPEED_POLL_URL.format(task_id=task_id)
        elapsed   = 0
        poll_data = {}
        print(f"⏳ {tag} Waiting for result (timeout: {WAVESPEED_TIMEOUT_S}s)...")
        while elapsed < WAVESPEED_TIMEOUT_S:
            time.sleep(WAVESPEED_POLL_DELAY)
            elapsed += WAVESPEED_POLL_DELAY
            try:
                poll_resp = requests.get(poll_url, headers=headers_auth, timeout=15)
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
            except requests.RequestException as e:
                print(f"⚠️  {tag} Polling error ({elapsed}s) : {e}")
                continue

            status = poll_data.get("data", {}).get("status", "")
            if status == "completed":
                outputs = poll_data.get("data", {}).get("outputs", [])
                if not outputs:
                    return self._handle_error(f"❌ {tag} Completed but no output.", _batch_idx)
                output_url = outputs[0]
                print(f"✅ {tag} Image ready! ({elapsed}s)")
                break
            elif status == "failed":
                error_msg = poll_data.get("data", {}).get("error", "Error inconnue")
                return self._handle_error(f"❌ {tag} Generation failed: {error_msg}", _batch_idx)
            else:
                print(f"   {tag} [{elapsed}s/{WAVESPEED_TIMEOUT_S}s] status={status!r}...")
        else:
            try:
                requests.delete(WAVESPEED_CANCEL_URL.format(task_id=task_id), headers=headers_auth, timeout=10)
            except Exception:
                pass
            return self._handle_error(f"❌ {tag} Timeout after {WAVESPEED_TIMEOUT_S}s.", _batch_idx)

        try:
            img_resp         = requests.get(output_url, timeout=60)
            img_resp.raise_for_status()
            pil_result       = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            image_np         = np.array(pil_result).astype(np.float32) / 255.0
            image_tensor_out = torch.from_numpy(image_np)[None,]
            timing_ms = poll_data.get("data", {}).get("timings", {}).get("inference", 0)
            info = (
                f"[WaveSpeed] Model: google/{model_label}\n"
                f"Resolution: {image_size} | Format: {output_format}\n"
                f"Inference time: {timing_ms}ms | Total: {elapsed}s"
            )
            print(f"🎉 {tag} Done!\n{info}")
            return (image_tensor_out, info, "")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Download error : {e}", _batch_idx)

    # ─────────────────────────────────────────────────────────────
    #  Helper URL publique
    #
    #  IMPORTANT pour les futures additions de modeles Kie :
    #  ----------------------------------------------------
    #  Quand tu ajoutes une nouvelle fonction _generate_<model>_kie(...) qui
    #  appelle ce helper pour uploader des images de reference :
    #
    #    1. Ajoute "kie_api_key" dans la signature de ta fonction (str = "")
    #    2. Passe-le au helper : self._tensor_to_public_url(
    #           tensor, idx=i,
    #           ws_api_key=ws_api_key,
    #           kie_api_key=kie_api_key,   # <-- ESSENTIEL pour les modeles Kie
    #       )
    #    3. Pour les modeles NON-Kie (Wavespeed/Fal/Vertex/...), NE PAS passer
    #       kie_api_key — laisse-le defaut "". Ils utiliseront la chaine
    #       de fallback classique (catbox/litterbox/etc.).
    #
    #  Pourquoi : avec kie_api_key, le helper essaie d'abord l'API d'upload de
    #  Kie (https://kieai.redpandaai.co/api/file-stream-upload). L'image est
    #  directement chez Kie -> evite les erreurs E9243 ("Director: unexpected
    #  error handling prediction") quand Kie n'arrive pas a fetcher une URL
    #  externe (catbox bloque par leur WAF, WaveSpeed CDN inaccessible, etc.).
    # ─────────────────────────────────────────────────────────────
    def _tensor_to_public_url(self, img_tensor, idx: int = 1, ws_api_key: str = "", kie_api_key: str = "") -> str | None:
        """Convert a tensor to a publicly fetchable URL via cascading uploads.

        Upload chain (priorite decroissante) :
          0. Kie file-stream-upload   (si kie_api_key fourni)        ← recommande pour les jobs Kie
          1. WaveSpeed CDN            (si ws_api_key fourni)
          2. catbox.moe               (anonymous)
          3. litterbox.catbox.moe     (anonymous, 1h TTL)
          4. 0x0.st                   (anonymous)
          5. telegra.ph               (anonymous)

        Returns the first successful URL, or None if all services fail.

        Compression auto : si le PNG depasse 10 MB (limite Kie file-upload),
        recodage en JPEG quality 100 -> 95 -> 90 -> ... -> 65 (pas de 5) jusqu'a passer sous 10 MB.
        Cible ~6 MB pour garder une bonne marge. Le buffer JPEG est ensuite
        utilise pour TOUS les services d'upload (plus rapide et plus fiable
        sur connexions instables).

        Args:
            img_tensor:    ComfyUI image tensor [1, H, W, 3] float 0..1
            idx:           Image index (1-based) — used for the filename
            ws_api_key:    WaveSpeed API key — active la fallback WaveSpeed CDN
            kie_api_key:   Kie API key — active l'upload Kie en priorite 0.
                           A passer UNIQUEMENT depuis les fonctions Kie ; pour
                           les autres providers (Wavespeed/Fal/Vertex/Google),
                           laisser defaut "".
        """
        pil_image = tensor_to_pil(img_tensor)
        if pil_image is None:
            return None
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format="PNG")

        # ── Compression JPEG conditionnelle (si > 10 MB, limite Kie) ──
        SIZE_LIMIT_BYTES = 10 * 1024 * 1024  # 10 MB
        png_size = img_buffer.tell()
        upload_filename = f"kie_input_{idx}.png"
        upload_mimetype = "image/png"

        if png_size > SIZE_LIMIT_BYTES:
            print(f"ℹ️  [Upload] Image {idx} PNG = {png_size / 1024 / 1024:.1f} MB > 10 MB limit → recoding to JPEG")
            # JPEG ne supporte pas RGBA → convert RGB d'abord
            rgb_image = pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image
            jpeg_buffer = None
            jpg_size = 0
            for q in (100, 95, 90, 85, 80, 75, 70, 65):
                jpeg_buffer = io.BytesIO()
                rgb_image.save(jpeg_buffer, format="JPEG", quality=q, optimize=True)
                jpg_size = jpeg_buffer.tell()
                if jpg_size <= SIZE_LIMIT_BYTES:
                    print(f"   [Upload] JPEG quality={q} → {jpg_size / 1024 / 1024:.2f} MB (was {png_size / 1024 / 1024:.1f} MB PNG)")
                    break
            else:
                # Aucune qualite n'a passe — on garde la derniere (Q=65).
                print(f"⚠️  [Upload] Image {idx} still {jpg_size / 1024 / 1024:.1f} MB at JPEG q=65 (uploads may fail).")
            img_buffer = jpeg_buffer
            upload_filename = f"kie_input_{idx}.jpg"
            upload_mimetype = "image/jpeg"

        _skip = NanoBananaAIO._failed_upload_services

        # --- Priorité 0 : Kie file-stream-upload (recommandé par Kie support) ---
        # Avantage : l'image est directement chez Kie, pas de fetch externe.
        # Évite les erreurs E9243 (Director: unexpected error handling prediction)
        # quand Kie n'arrive pas à fetcher une URL externe (catbox/WS/etc).
        # NOTE : on NE skip PAS Kie upload entre tentatives — c'est la methode
        # recommandee par leur support, on doit la tenter a chaque appel.
        if kie_api_key:
            img_buffer.seek(0)
            try:
                kie_resp = requests.post(
                    "https://kieai.redpandaai.co/api/file-stream-upload",
                    headers={"Authorization": f"Bearer {kie_api_key}"},
                    files={"file": (upload_filename, img_buffer, upload_mimetype)},
                    data={"uploadPath": "images/user-uploads"},
                    timeout=60,
                )
                kie_resp.raise_for_status()
                _kj = kie_resp.json()
                _url = (
                    (_kj.get("data") or {}).get("downloadUrl")
                    or _kj.get("downloadUrl")
                )
                if _url:
                    print(f"✅ [Upload] Image {idx} → Kie : {_url}")
                    return _url
                print(f"⚠️  [Upload] Kie response missing downloadUrl: {_kj} — falling back to other services for this image.")
            except Exception as e:
                print(f"⚠️  [Upload] Kie file-upload failed ({e}) — falling back to other services for this image.")

        # --- WaveSpeed CDN (requires ws_api_key) ---
        if ws_api_key and "wavespeed" not in _skip:
            img_buffer.seek(0)
            try:
                upload_resp = requests.post(
                    WAVESPEED_UPLOAD_URL,
                    headers={"Authorization": f"Bearer {ws_api_key}"},
                    files={"file": (upload_filename, img_buffer, upload_mimetype)},
                    timeout=60,
                )
                upload_resp.raise_for_status()
                url = upload_resp.json().get("data", {}).get("download_url")
                if url:
                    print(f"✅ [Upload] Image {idx} → OK")
                    return url
            except Exception as e:
                print(f"⚠️  [Upload] WaveSpeed CDN failed ({e}) — skipping for this session.")
                _skip.add("wavespeed")

        # --- Fallback 1 : catbox.moe ---
        if "catbox" not in _skip:
            img_buffer.seek(0)
            try:
                catbox_resp = requests.post(
                    "https://catbox.moe/user/api.php",
                    data={"reqtype": "fileupload"},
                    files={"fileToUpload": (upload_filename, img_buffer, upload_mimetype)},
                    timeout=30,
                )
                catbox_resp.raise_for_status()
                url = catbox_resp.text.strip()
                if url.startswith("https://"):
                    print(f"✅ [Upload] Image {idx} → catbox.moe : {url}")
                    return url
                _skip.add("catbox")
            except Exception as e:
                print(f"⚠️  [Upload] catbox.moe failed ({e}) — skipping for this session.")
                _skip.add("catbox")

        # --- Fallback 2 : litterbox.catbox.moe ---
        if "litterbox" not in _skip:
            img_buffer.seek(0)
            try:
                litter_resp = requests.post(
                    "https://litterbox.catbox.moe/resources/internals/api.php",
                    data={"reqtype": "fileupload", "time": "1h"},
                    files={"fileToUpload": (upload_filename, img_buffer, upload_mimetype)},
                    timeout=30,
                )
                litter_resp.raise_for_status()
                url = litter_resp.text.strip()
                if url.startswith("https://"):
                    print(f"✅ [Upload] Image {idx} → litterbox : {url}")
                    return url
                _skip.add("litterbox")
            except Exception as e:
                print(f"⚠️  [Upload] litterbox failed ({e}) — skipping for this session.")
                _skip.add("litterbox")

        # --- Fallback 3 : 0x0.st ---
        if "0x0" not in _skip:
            img_buffer.seek(0)
            try:
                zero_resp = requests.post(
                    "https://0x0.st",
                    files={"file": (upload_filename, img_buffer, upload_mimetype)},
                    timeout=30,
                )
                zero_resp.raise_for_status()
                url = zero_resp.text.strip()
                if url.startswith("https://"):
                    print(f"✅ [Upload] Image {idx} → 0x0.st : {url}")
                    return url
                _skip.add("0x0")
            except Exception as e:
                print(f"⚠️  [Upload] 0x0.st failed ({e}) — skipping for this session.")
                _skip.add("0x0")

        # --- Fallback 4 : telegra.ph ---
        if "telegraph" not in _skip:
            img_buffer.seek(0)
            try:
                tph_resp = requests.post(
                    "https://telegra.ph/upload",
                    files={"file": (upload_filename, img_buffer, upload_mimetype)},
                    timeout=30,
                )
                tph_resp.raise_for_status()
                tph_data = tph_resp.json()
                tph_path = tph_data[0].get("src", "") if isinstance(tph_data, list) else ""
                if tph_path.startswith("/"):
                    url = f"https://telegra.ph{tph_path}"
                    print(f"✅ [Upload] Image {idx} → telegra.ph : {url}")
                    return url
                _skip.add("telegraph")
            except Exception as e:
                print(f"⚠️  [Upload] telegra.ph failed ({e}) — skipping for this session.")
                _skip.add("telegraph")

        print(f"❌ [Upload] All upload services failed for image {idx}.")
        return None

    # ─────────────────────────────────────────────────────────────
    #  KIE.AI — NB Pro / NB2
    # ─────────────────────────────────────────────────────────────
    def _generate_kie(
        self,
        prompt, image_tensors, image_size, aspect_ratio,
        kie_api_key="", ws_api_key="", batch_size=1,
        is_nb2=False, negative_prompt="", _batch_idx=None,
    ):
        if not kie_api_key:
            return self._handle_error("❌ kie_api_key missing!", _batch_idx)

        kie_model   = "nano-banana-2" if is_nb2 else "nano-banana-pro"
        model_label = kie_model

        if batch_size and batch_size > 1:
            print(f"\n🔥 Parallel generation of {batch_size} images (Kie.ai)...\n")
            generated_images = []
            all_info         = []
            errors           = []
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {}
                for i in range(batch_size):
                    fut = executor.submit(
                        self._generate_kie,
                        prompt, image_tensors, image_size, aspect_ratio,
                        kie_api_key, ws_api_key, 1, is_nb2, negative_prompt, i + 1,
                    )
                    futures[fut] = i
                for future in as_completed(futures):
                    idx = futures[future] + 1
                    try:
                        result = future.result()
                        generated_images.append(result[0])
                        all_info.append(result[1])
                    except Exception as e:
                        errors.append(f"Batch {idx}: {e}")
                        print(f"❌ [KIE Batch {idx}] Failed — {e}")

            if not generated_images:
                return self._handle_error(
                    f"All Kie.ai generations failed: {'; '.join(errors)}"
                )
            ref_shape    = generated_images[0].shape
            valid_images = [img for img in generated_images if img.shape == ref_shape]
            if not valid_images:
                return self._handle_error("No valid image.")
            combined      = torch.cat(valid_images, dim=0)
            combined_info = "\n\n".join(all_info)
            if errors:
                combined_info = f"⚠️ {len(errors)} image(s) failed\n" + combined_info
            return (combined, combined_info, "")

        tag     = f"[KIE Batch {_batch_idx}]" if _batch_idx else "[Kie.ai]"
        headers = {"Authorization": f"Bearer {kie_api_key}", "Content-Type": "application/json"}

        image_urls = []
        if image_tensors:
            print(f"🖼️  {tag} Converting {len(image_tensors)} image(s)...")
            for i, tensor in enumerate(image_tensors, 1):
                url = self._tensor_to_public_url(tensor, idx=i, ws_api_key=ws_api_key, kie_api_key=kie_api_key)
                if url:
                    image_urls.append(url)
            # Fail-fast : aborte si une seule ref n'a pas pu etre uploadee
            if len(image_urls) < len(image_tensors):
                _missing = len(image_tensors) - len(image_urls)
                return self._handle_error(
                    f"❌ {tag} Upload failed: {len(image_urls)}/{len(image_tensors)} reference image(s) "
                    f"uploaded ({_missing} missing). Generation aborted to avoid producing an image without "
                    f"your references.",
                    _batch_idx,
                )

        payload = {
            "model": kie_model,
            "input": {
                "prompt":        prompt,
                "image_input":   image_urls,
                "aspect_ratio":  aspect_ratio,
                "resolution":    image_size,
                "output_format": "png",
            },
        }
        if negative_prompt and negative_prompt.strip():
            payload["input"]["negative_prompt"] = negative_prompt.strip()
        print(f"🚀 {tag} Submitting {model_label}...")

        # Option B: session locale par batch (evite stale connections du pool global)
        # Option C: jitter 0-0.5s avant POST en batch (evite burst simultane)
        _local_kie = _make_kie_session()
        if _batch_idx is not None:
            import random as _random_jitter
            time.sleep(_random_jitter.uniform(0, 0.5))
        try:
            resp = _local_kie.post(KIE_CREATE_URL, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            return self._handle_error(f"❌ {tag} Submission error : {e}", _batch_idx)

        if data.get("code") != 200:
            return self._handle_error(
                f"❌ {tag} Submission failed: {data.get('msg', 'unknown error')}", _batch_idx
            )
        task_id = data.get("data", {}).get("taskId")
        if not task_id:
            return self._handle_error(f"❌ {tag} No taskId: {data}", _batch_idx)
        print(f"🔖 {tag} Task ID : {task_id}")

        elapsed     = 0
        result_data = {}
        while elapsed < KIE_TIMEOUT_S:
            time.sleep(KIE_POLL_DELAY)
            elapsed += KIE_POLL_DELAY
            try:
                poll_resp = _local_kie.get(
                    KIE_POLL_URL, params={"taskId": task_id},
                    headers=headers, timeout=15,
                )
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
            except requests.RequestException as e:
                print(f"⚠️  {tag} Polling error ({elapsed}s) : {e}")
                continue

            if poll_data.get("code") != 200:
                return self._handle_error(f"❌ {tag} Poll error: {poll_data.get('msg')}", _batch_idx)
            result_data = poll_data.get("data", {})
            state       = result_data.get("state", "")
            if state == "success":
                print(f"✅ {tag} Task done! ({elapsed}s)")
                break
            elif state in ("fail", "failed", "error"):
                return self._handle_error(
                    f"❌ {tag} Failed: {result_data.get('failMsg', '?')}", _batch_idx
                )
            else:
                print(f"   {tag} [{elapsed}s/{KIE_TIMEOUT_S}s] state={state!r}...")
        else:
            return self._handle_error(f"❌ {tag} Timeout after {KIE_TIMEOUT_S}s.", _batch_idx)

        try:
            import json as _json
            result_json = _json.loads(result_data.get("resultJson", "{}"))
            result_urls = result_json.get("resultUrls", [])
            if not result_urls:
                return self._handle_error(f"❌ {tag} Aucune URL.", _batch_idx)
            output_url = result_urls[0]
        except Exception as e:
            return self._handle_error(f"❌ {tag} Parsing error : {e}", _batch_idx)

        try:
            img_resp         = requests.get(output_url, timeout=60)
            img_resp.raise_for_status()
            pil_result       = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            image_np         = np.array(pil_result).astype(np.float32) / 255.0
            image_tensor_out = torch.from_numpy(image_np)[None,]
            cost_ms = result_data.get("costTime", 0)
            info = (
                f"[Kie.ai] Model: {model_label}\n"
                f"Resolution: {image_size} | Ratio: {aspect_ratio}\n"
                f"Time: {cost_ms}ms | Total: {elapsed}s"
            )
            print(f"🎉 {tag} Done!\n{info}")
            return (image_tensor_out, info, "")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Download failed: {e}", _batch_idx)

    # ─────────────────────────────────────────────────────────────
    #  Auto-install OpenCV (headless)
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _ensure_opencv() -> bool:
        try:
            import cv2  # noqa: F401
            return True
        except ImportError:
            print("📦 [FaceSwap] Installing required dependency...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "opencv-python-headless", "--quiet",
                ])
                print("✅ [FaceSwap] Dependency installed successfully.")
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ [FaceSwap] Failed to install dependency: {e}")
                return False

    # ─────────────────────────────────────────────────────────────
    #  Mask face(s) in image tensor with a black rectangle (YuNet)
    # ─────────────────────────────────────────────────────────────
    def _mask_face_in_tensor(self, img_tensor, score_threshold=0.7):
        """
        Detects faces in the image 2 tensor and masks them
        with a black square (YuNet via OpenCV).
        Returns the modified tensor, or the original if OpenCV is unavailable.
        """
        if not self._ensure_opencv():
            print("⚠️  [FaceSwap] OpenCV unavailable — image 2 will not be masked")
            return img_tensor

        import cv2

        try:
            pil_img = tensor_to_pil(img_tensor)
            img_rgb = np.array(pil_img.convert("RGB"))
            img_bgr = img_rgb[:, :, ::-1].copy()
            h, w    = img_bgr.shape[:2]

            # Detection model — cached alongside the node to survive restarts
            model_dir  = os.path.join(os.path.dirname(__file__), ".yunet_cache")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "face_detection_yunet_2023mar.onnx")

            if not os.path.exists(model_path):
                print("📥 [FaceSwap] Downloading detection model for python...")
                _urllib_request.urlretrieve(
                    "https://github.com/opencv/opencv_zoo/raw/main/models/"
                    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
                    model_path,
                )
                print("✅ [FaceSwap] Detection model ready.")

            detector = cv2.FaceDetectorYN.create(
                model_path, "", (w, h),
                score_threshold=score_threshold, nms_threshold=0.3, top_k=5000,
            )
            min_face = min(w, h) * 0.05
            faces    = []

            _, detections = detector.detect(img_bgr)
            if detections is not None:
                for det in detections:
                    fx, fy, fw, fh = (
                        int(det[0]), int(det[1]), int(det[2]), int(det[3])
                    )
                    conf = float(det[-1])
                    if fw < min_face or fh < min_face:
                        continue
                    faces.append((fx, fy, fw, fh))
                    print(f"   [FaceSwap] Face detected ✅")

            if not faces:
                print("🔴 [FaceSwap] WARNING: No face detected in image 2 — face swap may fail!")
                return img_tensor

            for (fx, fy, fw, fh) in faces:
                pad_w = int(fw * 0.30)
                pad_h = int(fh * 0.30)
                x1 = max(0, fx - pad_w)
                y1 = max(0, fy - pad_h)
                x2 = min(w, fx + fw + pad_w)
                y2 = min(h, fy + fh + pad_h)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 0), -1)

            img_rgb_out = img_bgr[:, :, ::-1].copy()
            img_float   = img_rgb_out.astype(np.float32) / 255.0
            return torch.from_numpy(img_float)[None,]

        except Exception as exc:
            print(f"⚠️  [FaceSwap] Masking error: {exc} — using original image 2")
            return img_tensor

    # ─────────────────────────────────────────────────────────────
    #  Aspect ratio auto-detection for Face Swap Automation
    # ─────────────────────────────────────────────────────────────
    _NB_ASPECT_RATIOS = {
        "1:1":  1.0,
        "2:3":  2/3,
        "3:2":  3/2,
        "3:4":  3/4,
        "4:3":  4/3,
        "4:5":  4/5,
        "5:4":  5/4,
        "9:16": 9/16,
        "16:9": 16/9,
        "21:9": 21/9,
    }

    @staticmethod
    def _detect_aspect_ratio(img_tensor) -> str:
        """Return the Nano Banana ratio string closest to the actual image dimensions."""
        try:
            # img_tensor shape: [B, H, W, C]  (ComfyUI convention)
            h = img_tensor.shape[1]
            w = img_tensor.shape[2]
            if h == 0 or w == 0:
                return "1:1"
            actual = w / h
            best_key = min(
                NanoBananaAIO._NB_ASPECT_RATIOS,
                key=lambda k: abs(NanoBananaAIO._NB_ASPECT_RATIOS[k] - actual),
            )
            return best_key
        except Exception:
            return "1:1"

    # ─────────────────────────────────────────────────────────────
    #  Automation Face Swap — main dispatcher
    # ─────────────────────────────────────────────────────────────
    def _run_face_swap_automation(
        self,
        provider, model, image_size, aspect_ratio,
        batch_size, disable_safety, breast_refiner_enabled,
        low_neck_enabled,
        amateur_mode_enabled,
        face_expression,
        image_1, image_2,
        gemini_api_key="", wavespeed_api_key="", kie_api_key="",
        fal_api_key="", vertex_json_folder="", vertex_location="us-central1",
        temperature=1.0, top_p=0.95, fal_safety_tolerance="4",
        face_swap_custom_prompt="",
        yunet_score_threshold=0.7,
    ):
        if image_1 is None or image_2 is None:
            return self._handle_error(
                "❌ [Automation] Image 1 (reference face) "
                "and Image 2 (face to swap) are both required."
            )

        if model not in ("Nano Banana Pro", "Nano Banana 2", "Seedream 4.5"):
            model = "Nano Banana Pro"
            print("⚠️  [Automation] Incompatible model → forced: Nano Banana Pro")

        is_nb2       = (model == "Nano Banana 2")
        is_seedream  = (model == "Seedream 4.5")
        actual_model = _MODEL_MAP.get(model, "gemini-3-pro-image-preview")

        # Auto-detect aspect ratio from image_2 dimensions
        auto_ratio = self._detect_aspect_ratio(image_2)

        print(f"\n{'═'*55}")
        print("⚡ AUTOMATION FACE SWAP")
        print(f"   Provider     : {provider}")
        print(f"   Model        : {model}")
        print(f"   Aspect Ratio : {auto_ratio}  (auto-detected from image 2)")
        print(f"   Batch size   : {batch_size}")
        print(f"   Safety off   : {disable_safety}")
        print(f"   Breast Refin.: {'ON ✅' if breast_refiner_enabled else 'OFF'}")
        print(f"   Low Neck     : {'ON ✅' if low_neck_enabled else 'OFF'}")
        print(f"   Amateur Mode : {'ON ✅' if amateur_mode_enabled else 'OFF'}")
        print(f"{'═'*55}\n")

        # Step 1 : mask face in image_2
        print("🎭 [FaceSwap] Masking face in image 2...")
        masked_image_2 = self._mask_face_in_tensor(image_2, score_threshold=yunet_score_threshold)

        # Step 2 : decode hidden prompt (never printed to console)
        if amateur_mode_enabled:
            fs_prompt = base64.b64decode(_AMATEUR_MODE_PROMPT_B64).decode("utf-8")
        else:
            fs_prompt = base64.b64decode(_FACESWAP_PROMPT_B64).decode("utf-8")
        if breast_refiner_enabled:
            fs_prompt += "\n\n" + base64.b64decode(_BREAST_REFINER_B64).decode("utf-8")
        if low_neck_enabled:
            fs_prompt += "\n\nAdd a low neck"

        # Append facial expression instruction
        _EXPRESSION_MAP = {
            "Neutral":      "neutral facial expression",
            "Sensual":      "sensual facial expression, intelse eyes look",
            "Playful":      "playful facial expression",
            "Subtle smile": "subtle smile",
            "Smile":        "warm smile",
            "Laugh":        "she's laughing",
        }
        expression_text = _EXPRESSION_MAP.get(face_expression, "neutral facial expression")
        fs_prompt += f"\n\n{expression_text}"

        # Append user custom prompt (last, highest priority)
        if face_swap_custom_prompt and face_swap_custom_prompt.strip():
            fs_prompt += f"\n\n{face_swap_custom_prompt.strip()}"
            print(f"   ✏️  [FaceSwap] Custom prompt appended: \"{face_swap_custom_prompt.strip()[:80]}\"")

        image_tensors = [image_1, masked_image_2]

        # Safety: always OFF in face swap automation
        disable_safety      = True
        safety_threshold    = "OFF"
        fal_safety_tolerance = "6"

        print(f"🚀 [FaceSwap] Sending to {model} via {provider}...")

        # Step 3 : route to provider  (auto_ratio replaces aspect_ratio everywhere)
        if provider == "WAVESPEED":
            if is_seedream:
                return self._generate_seedream_wavespeed(
                    fs_prompt, image_tensors, image_size, auto_ratio,
                    batch_size=batch_size, ws_api_key=wavespeed_api_key,
                    disable_safety=disable_safety,
                )
            return self._generate_wavespeed(
                fs_prompt, image_tensors, image_size, "jpeg",
                auto_ratio, batch_size=batch_size,
                ws_api_key=wavespeed_api_key, is_nb2=is_nb2,
            )

        elif provider == "KIE":
            if is_seedream:
                return self._generate_seedream_kie(
                    fs_prompt, image_tensors, image_size, auto_ratio,
                    kie_api_key=kie_api_key, ws_api_key=wavespeed_api_key,
                    batch_size=batch_size, disable_safety=disable_safety,
                )
            return self._generate_kie(
                fs_prompt, image_tensors, image_size, auto_ratio,
                kie_api_key=kie_api_key, ws_api_key=wavespeed_api_key,
                batch_size=batch_size, is_nb2=is_nb2,
            )

        elif provider == "FAL":
            if is_seedream:
                return self._generate_seedream_fal(
                    fs_prompt, image_tensors, image_size, auto_ratio,
                    batch_size=batch_size, fal_api_key=fal_api_key,
                    disable_safety=disable_safety,
                )
            return self._generate_fal(
                prompt            = fs_prompt,
                image_tensors     = image_tensors,
                image_size        = image_size,
                aspect_ratio      = auto_ratio,
                batch_size        = batch_size,
                fal_api_key       = fal_api_key,
                is_nb2            = is_nb2,
                safety_tolerance  = fal_safety_tolerance,
                enable_web_search = False,
            )

        elif provider == "VERTEX":
            if is_seedream:
                return self._handle_error(
                    "❌ [FaceSwap] Seedream 4.5 is not supported with VERTEX provider. Use WAVESPEED, KIE or FAL."
                )
            if not vertex_json_folder:
                return self._handle_error(
                    "❌ [Automation] vertex_json_folder missing for VERTEX provider."
                )
            try:
                _vj_files = _load_vertex_json_folder(vertex_json_folder)
            except Exception as _e:
                return self._handle_error(str(_e))
            print(f"📁 [Vertex Automation] {len(_vj_files)} project(s) detected")
            if batch_size > len(_vj_files):
                print(f"⚠️  [Vertex Automation] batch_size={batch_size} > projects={len(_vj_files)} → batch reduced to {len(_vj_files)}")
                batch_size = len(_vj_files)
            # Rotation
            _off = NanoBananaAIO._vertex_rotation_offset % len(_vj_files)
            _vj_files = _vj_files[_off:] + _vj_files[:_off]
            NanoBananaAIO._vertex_rotation_offset = (_off + batch_size) % len(_vj_files)
            print(f"🔄 [Vertex Automation] Rotation offset={_off} → projects used : {[os.path.basename(f) for f in _vj_files[:batch_size]]}")
            if is_nb2:
                return self._generate_nb2_vertex(
                    prompt            = fs_prompt,
                    image_tensors     = image_tensors,
                    image_size        = image_size,
                    aspect_ratio      = auto_ratio,
                    temperature       = temperature,
                    vertex_json_files = _vj_files,
                    vertex_location   = vertex_location,
                    batch_size        = batch_size,
                    disable_safety    = disable_safety,
                )
            return self._generate_vertex(
                prompt              = fs_prompt,
                image_tensors       = image_tensors,
                image_size          = image_size,
                aspect_ratio        = auto_ratio,
                temperature         = temperature,
                top_p               = top_p,
                use_search          = False,
                system_instructions = None,
                batch_size          = batch_size,
                vertex_json_files   = _vj_files,
                vertex_location     = vertex_location,
                model_name          = actual_model,
                safety_threshold    = "BLOCK_NONE" if disable_safety else None,
            )

        else:  # GOOGLE
            if is_seedream:
                return self._handle_error(
                    "❌ [FaceSwap] Seedream 4.5 is not supported with GOOGLE provider. Use WAVESPEED, KIE or FAL."
                )
            if not gemini_api_key:
                return self._handle_error(
                    "❌ [Automation] gemini_api_key missing pour GOOGLE."
                )
            if is_nb2:
                return self._generate_nb2_google(
                    prompt         = fs_prompt,
                    image_tensors  = image_tensors,
                    image_size     = image_size,
                    aspect_ratio   = auto_ratio,
                    temperature    = temperature,
                    g_key          = gemini_api_key,
                    batch_size     = batch_size,
                    disable_safety = disable_safety,
                )
            approach = self._detect_approach(gemini_api_key, "", "")
            contents = [fs_prompt]
            for i, t in enumerate(image_tensors, 1):
                print(f"📷 [FaceSwap] Image {i} added to Google context...")
                contents.append(tensor_to_pil(t))
            if batch_size == 1:
                return self._generate_single_image(
                    actual_model, fs_prompt, False, approach, contents,
                    auto_ratio, image_size, temperature, top_p, None,
                    g_key=gemini_api_key, v_proj="", v_loc="",
                    safety_threshold=safety_threshold,
                )
            else:
                return self._generate_batch_parallel(
                    actual_model, fs_prompt, batch_size, False, approach, contents,
                    auto_ratio, image_size, temperature, top_p, None,
                    g_key=gemini_api_key, v_proj="", v_loc="",
                    safety_threshold=safety_threshold,
                )

    # ─────────────────────────────────────────────────────────────
    #  VIDEO — Point d'entrée principal
    # ─────────────────────────────────────────────────────────────
    def _generate_video(
        self,
        provider             = "GOOGLE",
        prompt               = "",
        model                = "Veo 3.1 Lite",
        video_resolution     = "1080p",
        aspect_ratio         = "16:9",
        duration             = 8,
        generate_audio       = True,
        gemini_api_key       = "",
        wavespeed_api_key    = "",
        kie_api_key          = "",
        fal_api_key          = "",
        fal_safety_tolerance = "4",
        vertex_json_folder   = "",
        vertex_location      = "us-central1",
        vertex_gcs_bucket    = "",
        image_tensors        = None,
        video_reference      = None,
        disable_safety       = False,
    ):
        """Routes video generation to the correct provider."""
        all_known = set(_VIDEO_MODEL_MAP.keys()) | _KLING_MODELS | _SEEDANCE_MODELS
        if model not in all_known:
            return self._handle_error(
                f"❌ [Video] Unknown video model: '{model}'.\n"
                f"Available models: {sorted(all_known)}"
            )

        # ── Snap de resolution ────────────────────────────────────
        # Seedance : 4K not supported (480p / 720p / 1080p OK)
        if model in _SEEDANCE_MODELS and video_resolution == "4K":
            print(f"⚠️  [Video] Seedance 2.0 ne supporte pas 4K → snap vers 1080p")
            video_resolution = "1080p"
        # 4K uniquement pour Kling 3.0 — les autres models snappent vers 1080p
        elif model != "Kling 3.0" and video_resolution == "4K":
            print(f"⚠️  [Video] {model} ne supporte pas 4K → snap vers 1080p")
            video_resolution = "1080p"
        # Autres models vidéo : pas de 480p
        elif model not in _SEEDANCE_MODELS and video_resolution == "480p":
            print(f"⚠️  [Video] {model} ne supporte pas 480p → snap vers 720p")
            video_resolution = "720p"

        # ── Kling 2.6 — KIE ──────────────────────────────────────
        if model == "Kling 2.6" and provider == "KIE":
            if not kie_api_key:
                return self._handle_error("❌ [Video] kie_api_key missing for Kling 2.6 / KIE.")
            return self._generate_kling26_kie(
                prompt         = prompt,
                aspect_ratio   = aspect_ratio,
                duration       = duration,
                generate_audio = generate_audio,
                kie_api_key    = kie_api_key,
                ws_api_key     = wavespeed_api_key,
                image_tensors  = image_tensors or [],
            )

        # ── Kling 2.6 — FAL (Pro 1080p uniquement) ───────────────
        if model == "Kling 2.6" and provider == "FAL":
            if not fal_api_key:
                return self._handle_error("❌ [Video] fal_api_key missing for Kling 2.6 / FAL.")
            return self._generate_kling26_fal(
                prompt         = prompt,
                aspect_ratio   = aspect_ratio,
                duration       = duration,
                resolution     = video_resolution,
                generate_audio = generate_audio,
                fal_api_key    = fal_api_key,
                image_tensors  = image_tensors or [],
            )

        # ── Kling 3.0 Motion Control — KIE ───────────────────────
        if model == "Kling 3.0 Motion Control" and provider == "KIE":
            if not kie_api_key:
                return self._handle_error("❌ [Video] kie_api_key missing for Kling 3.0 Motion Control / KIE.")
            return self._generate_kling_motion_control_kie(
                prompt          = prompt,
                resolution      = video_resolution,
                kie_api_key     = kie_api_key,
                ws_api_key      = wavespeed_api_key,
                image_tensors   = image_tensors or [],
                video_reference = video_reference,
            )

        # ── Kling 3.0 Motion Control — FAL ───────────────────────
        if model == "Kling 3.0 Motion Control" and provider == "FAL":
            if not fal_api_key:
                return self._handle_error("❌ [Video] fal_api_key missing for Kling 3.0 Motion Control / FAL.")
            return self._generate_kling_motion_control_fal(
                prompt          = prompt,
                resolution      = video_resolution,
                fal_api_key     = fal_api_key,
                image_tensors   = image_tensors or [],
                video_reference = video_reference,
            )

        # ── Kling 3.0 — FAL (Standard 720p / Pro 1080p / 4K) ────────
        if model == "Kling 3.0" and provider == "FAL":
            if not fal_api_key:
                return self._handle_error("❌ [Video] fal_api_key missing for Kling 3.0 / FAL.")
            return self._generate_kling30_fal(
                prompt         = prompt,
                aspect_ratio   = aspect_ratio,
                duration       = duration,
                resolution     = video_resolution,
                generate_audio = generate_audio,
                fal_api_key    = fal_api_key,
                image_tensors  = image_tensors or [],
            )

        # ── Kling 3.0 — KIE (Std 720p / Pro 1080p / 4K) ───────────────
        if model == "Kling 3.0" and provider == "KIE":
            if not kie_api_key:
                return self._handle_error("❌ [Video] kie_api_key missing for Kling 3.0 / KIE.")
            return self._generate_kling30_kie(
                prompt         = prompt,
                aspect_ratio   = aspect_ratio,
                duration       = duration,
                resolution     = video_resolution,
                generate_audio = generate_audio,
                kie_api_key    = kie_api_key,
                ws_api_key     = wavespeed_api_key,
                image_tensors  = image_tensors or [],
            )

        # ── Kling — WaveSpeed (Kling 3.0, 2.6, Motion Control) ───
        if model in _KLING_MODELS:
            if provider != "WAVESPEED":
                return self._handle_error(
                    f"❌ [Video] {model} est disponible via WAVESPEED, KIE ou FAL "
                    "(Kling 3.0 : WAVESPEED/KIE/FAL — Kling 2.6 : WAVESPEED/KIE/FAL — "
                    "Motion Control : WAVESPEED/FAL).\n"
                    "→ Select WAVESPEED, KIE or FAL."
                )
            if not wavespeed_api_key:
                return self._handle_error(
                    "❌ [Video] wavespeed_api_key missing pour Kling."
                )
            if model == "Kling 3.0 Motion Control":
                return self._generate_kling_motion_control_wavespeed(
                    prompt          = prompt,
                    resolution      = video_resolution,
                    ws_api_key      = wavespeed_api_key,
                    image_tensors   = image_tensors or [],
                    video_reference = video_reference,
                )
            return self._generate_kling_wavespeed(
                model          = model,
                prompt         = prompt,
                resolution     = video_resolution,
                duration       = duration,
                generate_audio = generate_audio,
                ws_api_key     = wavespeed_api_key,
                image_tensors  = image_tensors or [],
            )

        # ── Seedance 2.0 — WaveSpeed ──────────────────────────────
        if model in _SEEDANCE_MODELS and provider == "WAVESPEED":
            if not wavespeed_api_key:
                return self._handle_error("❌ [Video] wavespeed_api_key missing for Seedance 2.0 / WaveSpeed.")
            return self._generate_seedance_wavespeed(
                prompt         = prompt,
                aspect_ratio   = aspect_ratio,
                duration       = duration,
                resolution     = video_resolution,
                generate_audio = generate_audio,
                ws_api_key     = wavespeed_api_key,
                image_tensors  = image_tensors or [],
            )

        # ── Seedance 2.0 — FAL ────────────────────────────────────
        if model in _SEEDANCE_MODELS and provider == "FAL":
            if not fal_api_key:
                return self._handle_error("❌ [Video] fal_api_key missing for Seedance 2.0 / FAL.")
            return self._generate_seedance_fal(
                prompt         = prompt,
                aspect_ratio   = aspect_ratio,
                duration       = duration,
                resolution     = video_resolution,
                generate_audio = generate_audio,
                fal_api_key    = fal_api_key,
                image_tensors  = image_tensors or [],
            )

        # ── Seedance 2.0 — KIE ────────────────────────────────────
        if model in _SEEDANCE_MODELS and provider == "KIE":
            if not kie_api_key:
                return self._handle_error("❌ [Video] kie_api_key missing for Seedance 2.0 / KIE.")
            return self._generate_seedance_kie(
                prompt         = prompt,
                aspect_ratio   = aspect_ratio,
                duration       = duration,
                resolution     = video_resolution,
                generate_audio = generate_audio,
                kie_api_key    = kie_api_key,
                ws_api_key     = wavespeed_api_key,
                image_tensors  = image_tensors or [],
                disable_safety = disable_safety,
            )

        # ── Seedance 2.0 — Provider not supported ──────────────────
        if model in _SEEDANCE_MODELS:
            return self._handle_error(
                f"❌ [Video] Seedance 2.0 est disponible sur WAVESPEED, FAL et KIE uniquement.\n"
                f"→ Select WAVESPEED, FAL or KIE."
            )

        if provider == "GOOGLE":
            if not gemini_api_key:
                return self._handle_error(
                    "❌ [Video] gemini_api_key missing pour le provider GOOGLE."
                )
            return self._generate_veo_google(
                prompt         = prompt,
                model_id       = _VIDEO_MODEL_MAP[model],
                aspect_ratio   = aspect_ratio,
                duration       = duration,
                generate_audio = generate_audio,
                g_key          = gemini_api_key,
                image_tensors  = image_tensors or [],
            )
        elif provider == "VERTEX":
            if not vertex_json_folder:
                return self._handle_error(
                    "❌ [Video] vertex_json_folder missing for VERTEX provider."
                )
            try:
                _vj_files_vid = _load_vertex_json_folder(vertex_json_folder)
            except Exception as _e:
                return self._handle_error(str(_e))
            return self._generate_veo_vertex(
                prompt            = prompt,
                model_id          = _VIDEO_MODEL_MAP_VERTEX.get(model, _VIDEO_MODEL_MAP[model]),
                aspect_ratio      = aspect_ratio,
                duration          = duration,
                generate_audio    = generate_audio,
                vertex_json_path  = _vj_files_vid[0],
                vertex_location   = vertex_location,
                vertex_gcs_bucket = vertex_gcs_bucket,
                image_tensors     = image_tensors or [],
            )
        elif provider == "WAVESPEED":
            if not wavespeed_api_key:
                return self._handle_error(
                    "❌ [Video] wavespeed_api_key missing pour le provider WAVESPEED."
                )
            if model not in _VIDEO_MODEL_MAP_WAVESPEED:
                return self._handle_error(
                    f"❌ [Video] Le model '{model}' n'est pas disponible sur WaveSpeed.\n"
                    f"Available models: {list(_VIDEO_MODEL_MAP_WAVESPEED.keys())}"
                )
            return self._generate_veo_wavespeed(
                prompt         = prompt,
                model          = model,
                aspect_ratio   = aspect_ratio,
                duration       = duration,
                resolution     = video_resolution,
                generate_audio = generate_audio,
                ws_api_key     = wavespeed_api_key,
                image_tensors  = image_tensors or [],
            )
        elif provider == "KIE":
            if not kie_api_key:
                return self._handle_error(
                    "❌ [Video] kie_api_key missing pour le provider KIE."
                )
            if model not in _VIDEO_MODEL_MAP_KIE:
                return self._handle_error(
                    f"❌ [Video] Le model '{model}' n'est pas disponible sur Kie.ai.\n"
                    f"Available models: {list(_VIDEO_MODEL_MAP_KIE.keys())}"
                )
            return self._generate_veo_kie(
                prompt         = prompt,
                model          = model,
                aspect_ratio   = aspect_ratio,
                duration       = duration,
                resolution     = video_resolution,
                generate_audio = generate_audio,
                kie_api_key    = kie_api_key,
                ws_api_key     = wavespeed_api_key,
                image_tensors  = image_tensors or [],
            )
        elif provider == "FAL":
            if not fal_api_key:
                return self._handle_error(
                    "❌ [Video] fal_api_key missing pour le provider FAL."
                )
            if model not in _VIDEO_MODEL_MAP_FAL:
                return self._handle_error(
                    f"❌ [Video] Le model '{model}' n'est pas disponible sur FAL.\n"
                    f"Available models: {list(_VIDEO_MODEL_MAP_FAL.keys())}"
                )
            return self._generate_veo_fal(
                prompt           = prompt,
                model            = model,
                aspect_ratio     = aspect_ratio,
                duration         = duration,
                resolution       = video_resolution,
                generate_audio   = generate_audio,
                fal_api_key      = fal_api_key,
                safety_tolerance = fal_safety_tolerance,
                image_tensors    = image_tensors or [],
            )
        else:
            return self._handle_error(
                f"❌ [Video] Provider '{provider}' does not support Veo video generation.\n"
                "→ Select GOOGLE, VERTEX, WAVESPEED, KIE or FAL to use Veo models."
            )

    # ─────────────────────────────────────────────────────────────
    #  VIDEO — Google AI Studio (Gemini API)
    # ─────────────────────────────────────────────────────────────
    def _generate_veo_google(
        self,
        prompt         = "",
        model_id       = "veo-3.0-generate-preview",
        aspect_ratio   = "16:9",
        duration       = 8,
        generate_audio = True,
        g_key          = "",
        image_tensors  = None,
    ):
        tag = "[Veo Google]"
        print(f"🎬 {tag} Starting — model={model_id} | ratio={aspect_ratio} | duration={duration}s")

        try:
            client = genai.Client(api_key=g_key)
        except Exception as e:
            return self._handle_error(f"❌ {tag} Unable to create Google client: {e}")

        duration_clamped = max(4, min(8, int(duration)))
        if duration_clamped != duration:
            print(f"⚠️  {tag} duration {duration}s out of API range (4-8s), adjusted to {duration_clamped}s")

        config = types.GenerateVideosConfig(
            aspect_ratio     = aspect_ratio,
            duration_seconds = duration_clamped,
            generate_audio   = generate_audio,
        )

        try:
            if image_tensors:
                pil_img = tensor_to_pil(image_tensors[0])
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                buf.seek(0)
                src_image = types.Image(
                    image_bytes = buf.getvalue(),
                    mime_type   = "image/png",
                )
                print(f"🖼️  {tag} Image-to-video mode (image_1 used as reference)")
                operation = client.models.generate_videos(
                    model  = model_id,
                    prompt = prompt,
                    image  = src_image,
                    config = config,
                )
            else:
                print(f"✏️  {tag} Mode text-to-video")
                operation = client.models.generate_videos(
                    model  = model_id,
                    prompt = prompt,
                    config = config,
                )
        except Exception as e:
            return self._handle_error(
                f"❌ {tag} Error starting generation: {type(e).__name__}: {e}"
            )

        print(f"⏳ {tag} Generating (may take several minutes)...")
        elapsed = 0
        while elapsed < _VIDEO_TIMEOUT_S:
            try:
                if operation.done:
                    break
            except Exception:
                pass
            time.sleep(_VIDEO_POLL_DELAY)
            elapsed += _VIDEO_POLL_DELAY
            try:
                operation = client.operations.get(operation)
                print(f"   {tag} [{elapsed}s/{_VIDEO_TIMEOUT_S}s] waiting...")
            except Exception as e:
                print(f"⚠️  {tag} Polling error ({elapsed}s) : {e}")
        else:
            return self._handle_error(f"❌ {tag} Timeout after {_VIDEO_TIMEOUT_S}s.")

        try:
            # Veo 3.0 → .result  |  Veo 3.1 → .response
            try:
                generated = operation.result.generated_videos
            except Exception:
                generated = operation.response.generated_videos

            if not generated:
                return self._handle_error(f"❌ {tag} No video returned by API.")

            generated_video = generated[0]
            # CRITIQUE : files.download() OBLIGATOIRE avant d'accéder à video_bytes
            client.files.download(file=generated_video.video)
            video_bytes = generated_video.video.video_bytes

            if not video_bytes:
                return self._handle_error(
                    f"❌ {tag} video_bytes empty after download — check Files API permissions."
                )
        except Exception as e:
            return self._handle_error(
                f"❌ {tag} Unable to read video response: {type(e).__name__}: {e}"
            )

        return self._finalize_video(video_bytes, tag, model_id, aspect_ratio, duration)

    # ─────────────────────────────────────────────────────────────
    #  VIDEO — Vertex AI
    # ─────────────────────────────────────────────────────────────
    def _generate_veo_vertex(
        self,
        prompt            = "",
        model_id          = "veo-3.1-lite-generate-001",
        aspect_ratio      = "16:9",
        duration          = 8,
        generate_audio    = True,
        vertex_json_path  = "",
        vertex_location   = "us-central1",
        vertex_gcs_bucket = "",
        image_tensors     = None,
    ):
        """Vertex AI video generation via REST (predictLongRunning + fetchPredictOperation)."""
        tag      = "[Veo Vertex]"
        location = vertex_location or "us-central1"
        print(f"🎬 {tag} Starting — model={model_id} | ratio={aspect_ratio} | duration={duration}s")

        # ── 1. Credentials & token ────────────────────────────────
        try:
            import google.auth.transport.requests as _gtr
            credentials, project_id = _load_vertex_credentials(vertex_json_path)
            credentials.refresh(_gtr.Request())
            token = credentials.token
            print(f"🌐 {tag} Projet={project_id} | location={location}")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Invalid credentials: {e}")

        # ── 2. GCS bucket (optionnel — Vertex auto-crée gs://veo-videos-{project}) ──
        # Défense contre True/None/bool que ComfyUI peut envoyer pour un input vide
        _raw_gcs   = vertex_gcs_bucket if isinstance(vertex_gcs_bucket, str) else ""
        gcs_bucket = _raw_gcs.strip()
        print(f"📦 {tag} vertex_gcs_bucket received: {repr(vertex_gcs_bucket)} → gcs_bucket={repr(gcs_bucket)}")

        # ── 3. Construction de la requête ─────────────────────────
        base     = f"https://{location}-aiplatform.googleapis.com/v1"
        model_path = (
            f"projects/{project_id}/locations/{location}"
            f"/publishers/google/models/{model_id}"
        )
        headers = {
            "Authorization":      f"Bearer {token}",
            "Content-Type":       "application/json; charset=utf-8",
            "x-goog-user-project": project_id,
        }

        instance: dict = {"prompt": prompt}
        if image_tensors:
            try:
                pil_img = tensor_to_pil(image_tensors[0])
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                instance["image"] = {
                    "bytesBase64Encoded": base64.b64encode(buf.getvalue()).decode(),
                    "mimeType": "image/png",
                }
                print(f"🖼️  {tag} Mode image-to-video")
            except Exception as e:
                print(f"⚠️  {tag} Unable to encode image: {e}")

        params: dict = {
            "sampleCount":     1,
            "durationSeconds": duration,
            "aspectRatio":     aspect_ratio,
            "generateAudio":   generate_audio,
        }
        # N'ajouter storageUri que si c'est un vrai URI gs:// — Vertex auto-bucket sinon
        if gcs_bucket.startswith("gs://"):
            params["storageUri"] = gcs_bucket
            print(f"📦 {tag} storageUri sent: {gcs_bucket}")
        else:
            print(f"📦 {tag} storageUri omis — Vertex utilisera son bucket auto")

        payload = {
            "instances":  [instance],
            "parameters": params,
        }

        # ── 4. Lancement de l'opération longue ────────────────────
        try:
            resp = requests.post(
                f"{base}/{model_path}:predictLongRunning",
                headers = headers,
                json    = payload,
                timeout = 30,
            )
            resp.raise_for_status()
            operation_name = resp.json()["name"]
            print(f"✅ {tag} Operation started: {operation_name}")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Launch error : {e}")

        # ── 5. Polling fetchPredictOperation ──────────────────────
        print(f"⏳ {tag} Generating...")
        elapsed = 0
        result  = None
        while elapsed < _VIDEO_TIMEOUT_S:
            time.sleep(_VIDEO_POLL_DELAY)
            elapsed += _VIDEO_POLL_DELAY
            try:
                fetch = requests.post(
                    f"{base}/{model_path}:fetchPredictOperation",
                    headers = headers,
                    json    = {"operationName": operation_name},
                    timeout = 30,
                )
                data = fetch.json()
                status = "✅ done" if data.get("done") else "⏳ pending"
                print(f"   {tag} [{elapsed}s/{_VIDEO_TIMEOUT_S}s] {status}")
                if data.get("done"):
                    result = data
                    break
            except Exception as e:
                print(f"⚠️  {tag} Polling error ({elapsed}s) : {e}")
        else:
            return self._handle_error(f"❌ {tag} Timeout after {_VIDEO_TIMEOUT_S}s.")

        # ── 6. Parsing de la réponse ──────────────────────────────
        try:
            response_body = result.get("response", {})
            videos = (
                response_body.get("videos")
                or response_body.get("generatedSamples")
                or []
            )
            if not videos:
                return self._handle_error(
                    f"❌ {tag} No video in response.\n"
                    f"Full response: {result}"
                )
            video_entry = videos[0]
            # Log la structure pour debug futur
            print(f"📼 {tag} Structure video_entry : { {k: (v[:40]+'…' if isinstance(v,str) and len(v)>40 else v) for k,v in video_entry.items()} }")

            # Cas 1 : vidéo retournée en base64 directement (sans storageUri)
            b64 = (
                video_entry.get("bytesBase64Encoded")
                or video_entry.get("video", {}).get("bytesBase64Encoded", "")
            )
            if b64:
                video_bytes = base64.b64decode(b64)
                print(f"✅ {tag} Video decoded from base64 ({len(video_bytes)} bytes).")
                return self._finalize_video(video_bytes, tag, model_id, aspect_ratio, duration)

            # Cas 2 : vidéo stockée dans GCS
            gcs_uri = (
                video_entry.get("gcsUri")
                or video_entry.get("video", {}).get("gcsUri", "")
            )
            if not gcs_uri:
                return self._handle_error(
                    f"❌ {tag} Neither base64 nor gcsUri in response.\n"
                    f"video_entry = {video_entry}"
                )
            print(f"📼 {tag} GCS video: {gcs_uri}")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Response read : {e}")

        # ── 7. Download depuis GCS (Storage REST API) ───────
        try:
            gcs_path    = gcs_uri.removeprefix("gs://")
            bucket_name, _, object_name = gcs_path.partition("/")
            encoded_obj = requests.utils.quote(object_name, safe="")
            dl_url      = (
                f"https://storage.googleapis.com/storage/v1/b"
                f"/{bucket_name}/o/{encoded_obj}?alt=media"
            )
            dl_resp = requests.get(
                dl_url,
                headers = {"Authorization": f"Bearer {token}"},
                timeout = 120,
            )
            dl_resp.raise_for_status()
            video_bytes = dl_resp.content
            print(f"✅ {tag} Video downloaded from GCS ({len(video_bytes)} bytes).")
        except Exception as e:
            return self._handle_error(
                f"❌ {tag} Unable to download from GCS ({gcs_uri}): {e}"
            )

        return self._finalize_video(video_bytes, tag, model_id, aspect_ratio, duration)

    # ─────────────────────────────────────────────────────────────
    #  VIDEO — WaveSpeed Kling (Standard / Pro)
    # ─────────────────────────────────────────────────────────────
    def _generate_kling_wavespeed(
        self,
        model          = "Kling 3.0",
        prompt         = "",
        resolution     = "720p",
        duration       = 5,
        generate_audio = True,
        ws_api_key     = "",
        image_tensors  = None,
    ):
        # 720p → Standard, 1080p → Pro (fallback sur Standard)
        model_map = _KLING_WS_URL_MAP.get(model, _KLING_WS_URL_MAP["Kling 3.0"])
        slug    = model_map.get(resolution, model_map["720p"])
        variant = "4K" if resolution == "4K" else ("Pro" if resolution == "1080p" else "Standard")
        url     = f"{WAVESPEED_BASE_URL}/{slug}"
        tag     = f"[WaveSpeed {model} {variant}]"
        headers = {
            "Authorization": f"Bearer {ws_api_key}",
            "Content-Type":  "application/json",
        }

        # Kling 2.6 n'accepte que 5 ou 10s ; Kling 3.0 accepte 3-15s
        if model == "Kling 2.6":
            dur = 10 if int(duration) > 7 else 5
        else:
            dur = max(3, min(15, int(duration)))
        print(f"🎬 {tag} Initializing — duration={dur}s | audio={generate_audio}")

        # ── 1. Upload de l'image de référence (optionnel) ─────────
        image_url = None
        if image_tensors:
            image_url = self._tensor_to_public_url(image_tensors[0], idx=1, ws_api_key=ws_api_key)
            if image_url:
                print(f"🖼️  {tag} Image uploaded → {image_url[:60]}…")
            else:
                print(f"⚠️  {tag} Image upload failed — text-to-video mode.")

        # ── 2. Payload ────────────────────────────────────────────
        payload: dict = {
            "prompt":       prompt,
            "duration":     dur,
            "cfg_scale":    0.5,
            "shot_type":    "customize",
            "element_list": [],
            "multi_prompt": [],
            "sound":        generate_audio,
        }
        if image_url:
            payload["image"] = image_url

        # ── 3. Soumission ─────────────────────────────────────────
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            task_id = (data.get("data") or {}).get("id") or data.get("id")
            if not task_id:
                return self._handle_error(f"❌ {tag} Submission failed: {data}")
            print(f"🔖 {tag} Task ID : {task_id}")
        except requests.RequestException as e:
            return self._handle_error(f"❌ {tag} Submission error : {e}")

        # ── 4. Polling ────────────────────────────────────────────
        poll_url  = WAVESPEED_POLL_URL.format(task_id=task_id)
        elapsed   = 0
        poll_data = {}
        print(f"⏳ {tag} Waiting for result (timeout: {_VIDEO_TIMEOUT_S}s)...")
        while elapsed < _VIDEO_TIMEOUT_S:
            time.sleep(_VIDEO_POLL_DELAY)
            elapsed += _VIDEO_POLL_DELAY
            try:
                poll_resp = requests.get(poll_url, headers={"Authorization": f"Bearer {ws_api_key}"}, timeout=15)
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
            except requests.RequestException as e:
                print(f"⚠️  {tag} Polling error ({elapsed}s) : {e}")
                continue

            status = poll_data.get("data", {}).get("status", "")
            if status == "completed":
                outputs = poll_data.get("data", {}).get("outputs", [])
                if not outputs:
                    return self._handle_error(f"❌ {tag} Completed but no video output.")
                video_url = outputs[0]
                print(f"✅ {tag} Video ready! ({elapsed}s)")
                break
            elif status == "failed":
                error_msg = poll_data.get("data", {}).get("error", "Error inconnue")
                return self._handle_error(f"❌ {tag} Generation failed: {error_msg}")
            else:
                print(f"   {tag} [{elapsed}s/{_VIDEO_TIMEOUT_S}s] status={status!r}...")
        else:
            try:
                requests.delete(
                    WAVESPEED_CANCEL_URL.format(task_id=task_id),
                    headers={"Authorization": f"Bearer {ws_api_key}"},
                    timeout=10,
                )
            except Exception:
                pass
            return self._handle_error(f"❌ {tag} Timeout after {_VIDEO_TIMEOUT_S}s.")

        # ── 5. Download ─────────────────────────────────────
        try:
            dl_resp = requests.get(video_url, timeout=120)
            dl_resp.raise_for_status()
            video_bytes = dl_resp.content
            print(f"✅ {tag} Video downloaded ({len(video_bytes)} bytes).")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Download error : {e}")

        return self._finalize_video(video_bytes, tag, slug, "16:9", dur)

    # ─────────────────────────────────────────────────────────────
    #  Kling 3.0 Motion Control — WaveSpeed
    # ─────────────────────────────────────────────────────────────
    def _video_to_public_url(self, video_ref: str, ws_api_key: str = "", kie_api_key: str = "") -> str | None:
        """Uploads a local video and returns a public URL.

        Upload chain (priority order):
          0. Kie file-stream-upload  (if kie_api_key provided)  ← no external fetch needed
          1. WaveSpeed binary upload (if ws_api_key provided)
        Falls back to next option on failure.
        If video_ref is already a URL (http/https), it is returned as-is.
        """
        if not video_ref:
            return None
        video_ref = video_ref.strip()
        if video_ref.startswith(("http://", "https://")):
            return video_ref
        if not os.path.isfile(video_ref):
            print(f"⚠️  [MotionControl] Video file not found: {video_ref}")
            return None

        ext      = os.path.splitext(video_ref)[1].lower() or ".mp4"
        mime     = "video/mp4" if ext == ".mp4" else f"video/{ext.lstrip('.')}"
        filename = os.path.basename(video_ref)

        with open(video_ref, "rb") as f:
            video_bytes = f.read()

        # --- Priority 0 : Kie file-stream-upload ---
        if kie_api_key:
            try:
                kie_resp = requests.post(
                    "https://kieai.redpandaai.co/api/file-stream-upload",
                    headers={"Authorization": f"Bearer {kie_api_key}"},
                    files={"file": (filename, video_bytes, mime)},
                    data={"uploadPath": "videos/user-uploads"},
                    timeout=120,
                )
                kie_resp.raise_for_status()
                _kj  = kie_resp.json()
                _url = (_kj.get("data") or {}).get("downloadUrl") or _kj.get("downloadUrl")
                if _url:
                    print(f"✅ [Upload] Video → Kie : {_url[:70]}…")
                    return _url
                print(f"⚠️  [Upload] Kie video-upload missing downloadUrl: {_kj} — trying WaveSpeed.")
            except Exception as e:
                print(f"⚠️  [Upload] Kie video-upload failed ({e}) — trying WaveSpeed.")

        # --- Priority 1 : WaveSpeed binary upload ---
        if ws_api_key:
            try:
                resp = requests.post(
                    WAVESPEED_UPLOAD_URL,
                    headers={"Authorization": f"Bearer {ws_api_key}"},
                    files={"file": (filename, video_bytes, mime)},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                url  = (data.get("data") or {}).get("download_url") or (data.get("data") or {}).get("url")
                if url:
                    print(f"✅ [Upload] Video → WaveSpeed : {url[:70]}…")
                    return url
                print(f"⚠️  [Upload] WaveSpeed video-upload missing url: {data}")
            except Exception as e:
                print(f"⚠️  [Upload] WaveSpeed video-upload failed ({e})")

        print("❌ [Upload] All video upload methods failed.")
        return None

    def _generate_kling_motion_control_wavespeed(
        self,
        prompt          = "",
        resolution      = "720p",
        ws_api_key      = "",
        image_tensors   = None,
        video_reference = None,
    ):
        variant = "Pro" if resolution == "1080p" else "Std"
        tag     = f"[WaveSpeed Kling 3.0 Motion Control {variant}]"

        if not video_reference:
            return self._handle_error(
                f"❌ {tag} video_reference missing.\n"
                "→ Connect a reference video to the 'video_reference' input."
            )

        slug_map = _KLING_WS_URL_MAP["Kling 3.0 Motion Control"]
        slug     = slug_map.get(resolution, slug_map["720p"])
        url      = f"{WAVESPEED_BASE_URL}/{slug}"
        headers  = {
            "Authorization": f"Bearer {ws_api_key}",
            "Content-Type":  "application/json",
        }

        print(f"🎬 {tag} Initialisation")

        # ── 1. Upload de l'image de référence (optionnel) ─────────
        image_url = None
        if image_tensors:
            image_url = self._tensor_to_public_url(image_tensors[0], idx=1, ws_api_key=ws_api_key)
            if image_url:
                print(f"🖼️  {tag} Image uploaded → {image_url[:60]}…")
            else:
                print(f"⚠️  {tag} Image upload failed — will be skipped.")

        # ── 2. Upload de la vidéo de référence ────────────────────
        print(f"🎞️  {tag} Uploading reference video...")
        video_url = self._video_to_public_url(video_reference, ws_api_key=ws_api_key)
        if not video_url:
            return self._handle_error(
                f"❌ {tag} Unable to upload reference video."
            )
        print(f"🎞️  {tag} Reference video → {video_url[:60]}…")

        # ── 3. Payload ────────────────────────────────────────────
        payload: dict = {
            "prompt":                prompt,
            "video":                 video_url,
            "character_orientation": "video",
            "element_list":          [],
            "keep_original_sound":   True,
            "shot_type":             "customize",
        }
        if image_url:
            payload["image"] = image_url

        # ── 4. Soumission ─────────────────────────────────────────
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data    = resp.json()
            task_id = (data.get("data") or {}).get("id") or data.get("id")
            if not task_id:
                return self._handle_error(f"❌ {tag} Submission failed: {data}")
            print(f"🔖 {tag} Task ID : {task_id}")
        except requests.RequestException as e:
            return self._handle_error(f"❌ {tag} Submission error : {e}")

        # ── 5. Polling ────────────────────────────────────────────
        poll_url  = WAVESPEED_POLL_URL.format(task_id=task_id)
        elapsed   = 0
        poll_data = {}
        print(f"⏳ {tag} Waiting for result (timeout: {_MOTION_CONTROL_TIMEOUT_S}s)...")
        while elapsed < _MOTION_CONTROL_TIMEOUT_S:
            time.sleep(_VIDEO_POLL_DELAY)
            elapsed += _VIDEO_POLL_DELAY
            try:
                poll_resp = requests.get(
                    poll_url,
                    headers={"Authorization": f"Bearer {ws_api_key}"},
                    timeout=15,
                )
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
            except requests.RequestException as e:
                print(f"⚠️  {tag} Polling error ({elapsed}s) : {e}")
                continue

            status = poll_data.get("data", {}).get("status", "")
            if status == "completed":
                outputs = poll_data.get("data", {}).get("outputs", [])
                if not outputs:
                    return self._handle_error(f"❌ {tag} Completed but no video output.")
                video_out_url = outputs[0]
                print(f"✅ {tag} Video ready! ({elapsed}s)")
                break
            elif status == "failed":
                error_msg = poll_data.get("data", {}).get("error", "Error inconnue")
                return self._handle_error(f"❌ {tag} Generation failed: {error_msg}")
            else:
                print(f"   {tag} [{elapsed}s/{_MOTION_CONTROL_TIMEOUT_S}s] status={status!r}...")
        else:
            try:
                requests.delete(
                    WAVESPEED_CANCEL_URL.format(task_id=task_id),
                    headers={"Authorization": f"Bearer {ws_api_key}"},
                    timeout=10,
                )
            except Exception:
                pass
            return self._handle_error(f"❌ {tag} Timeout after {_MOTION_CONTROL_TIMEOUT_S}s.")

        # ── 6. Download ─────────────────────────────────────
        try:
            dl_resp     = requests.get(video_out_url, timeout=120)
            dl_resp.raise_for_status()
            video_bytes = dl_resp.content
            print(f"✅ {tag} Video downloaded ({len(video_bytes)} bytes).")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Download error : {e}")

        return self._finalize_video(video_bytes, tag, slug, "16:9", 0)

    # ─────────────────────────────────────────────────────────────
    #  VIDEO — WaveSpeed (Veo 3.1 / Fast / Lite)
    # ─────────────────────────────────────────────────────────────
    def _generate_veo_wavespeed(
        self,
        prompt         = "",
        model          = "Veo 3.1 Lite",
        aspect_ratio   = "16:9",
        duration       = 8,
        resolution     = "1080p",
        generate_audio = True,
        ws_api_key     = "",
        image_tensors  = None,
    ):
        slug    = _VIDEO_MODEL_MAP_WAVESPEED[model]
        tag     = f"[WaveSpeed {model}]"
        url     = f"{WAVESPEED_BASE_URL}/google/{slug}/image-to-video"
        headers = {
            "Authorization": f"Bearer {ws_api_key}",
            "Content-Type":  "application/json",
        }
        print(f"🎬 {tag} Starting — ratio={aspect_ratio} | duration={duration}s | resolution={resolution}")

        # ── 1. Upload de l'image de référence (optionnel) ─────────
        image_url = None
        if image_tensors:
            image_url = self._tensor_to_public_url(image_tensors[0], idx=1, ws_api_key=ws_api_key)
            if image_url:
                print(f"🖼️  {tag} Image uploaded → {image_url[:60]}…")
            else:
                print(f"⚠️  {tag} Image upload failed, text-to-video mode.")

        # ── 2. Soumission ─────────────────────────────────────────
        duration_clamped = max(4, min(8, int(duration)))
        if duration_clamped != duration:
            print(f"⚠️  {tag} duration {duration}s adjusted to {duration_clamped}s (range 4-8)")

        payload = {
            "aspect_ratio":   aspect_ratio,
            "duration":       duration_clamped,
            "generate_audio": generate_audio,
            "prompt":         prompt,
            "resolution":     resolution,
        }
        if image_url:
            payload["image"] = image_url

        try:
            submit_resp = requests.post(url, json=payload, headers=headers, timeout=30)
            submit_resp.raise_for_status()
            submit_data = submit_resp.json()
            task_id = (submit_data.get("data") or {}).get("id") or submit_data.get("id")
            if not task_id:
                return self._handle_error(f"❌ {tag} Submission failed: {submit_data}")
            print(f"🔖 {tag} Task ID : {task_id}")
        except requests.RequestException as e:
            return self._handle_error(f"❌ {tag} Submission error : {e}")

        # ── 3. Polling ────────────────────────────────────────────
        poll_url = WAVESPEED_POLL_URL.format(task_id=task_id)
        elapsed  = 0
        poll_data = {}
        print(f"⏳ {tag} Waiting for result (timeout: {_VIDEO_TIMEOUT_S}s)...")
        while elapsed < _VIDEO_TIMEOUT_S:
            time.sleep(_VIDEO_POLL_DELAY)
            elapsed += _VIDEO_POLL_DELAY
            try:
                poll_resp = requests.get(poll_url, headers={"Authorization": f"Bearer {ws_api_key}"}, timeout=15)
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
            except requests.RequestException as e:
                print(f"⚠️  {tag} Polling error ({elapsed}s) : {e}")
                continue

            status = poll_data.get("data", {}).get("status", "")
            if status == "completed":
                outputs = poll_data.get("data", {}).get("outputs", [])
                if not outputs:
                    return self._handle_error(f"❌ {tag} Completed but no video output.")
                video_url = outputs[0]
                print(f"✅ {tag} Video ready! ({elapsed}s)")
                break
            elif status == "failed":
                error_msg = poll_data.get("data", {}).get("error", "Error inconnue")
                return self._handle_error(f"❌ {tag} Generation failed: {error_msg}")
            else:
                print(f"   {tag} [{elapsed}s/{_VIDEO_TIMEOUT_S}s] status={status!r}...")
        else:
            try:
                requests.delete(
                    WAVESPEED_CANCEL_URL.format(task_id=task_id),
                    headers={"Authorization": f"Bearer {ws_api_key}"},
                    timeout=10,
                )
            except Exception:
                pass
            return self._handle_error(f"❌ {tag} Timeout after {_VIDEO_TIMEOUT_S}s.")

        # ── 4. Download de la vidéo ─────────────────────────
        try:
            dl_resp = requests.get(video_url, timeout=120)
            dl_resp.raise_for_status()
            video_bytes = dl_resp.content
            print(f"✅ {tag} Video downloaded ({len(video_bytes)} bytes).")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Video download error : {e}")

        return self._finalize_video(video_bytes, tag, slug, aspect_ratio, duration_clamped)

    # ─────────────────────────────────────────────────────────────
    #  VIDEO — Kie.ai (Veo 3.1 / Fast / Lite)
    # ─────────────────────────────────────────────────────────────
    def _generate_veo_kie(
        self,
        prompt         = "",
        model          = "Veo 3.1 Lite",
        aspect_ratio   = "16:9",
        duration       = 8,
        resolution     = "1080p",
        generate_audio = True,
        kie_api_key    = "",
        ws_api_key     = "",
        image_tensors  = None,
        _batch_idx     = None,
    ):
        kie_model = _VIDEO_MODEL_MAP_KIE[model]
        tag       = f"[Kie.ai {model}]"
        headers   = {
            "Authorization": f"Bearer {kie_api_key}",
            "Content-Type":  "application/json",
        }
        print(f"🎬 {tag} Initialisation — ratio={aspect_ratio} | resolution={resolution}")
        if not generate_audio:
            print(f"ℹ️  {tag} generate_audio=False ignored — Kie.ai always includes audio.")

        # ── 1. Upload des images de référence (optionnel) ─────────
        image_urls = []
        for idx, tensor in enumerate(image_tensors or []):
            url = self._tensor_to_public_url(tensor, idx=idx + 1, ws_api_key=ws_api_key, kie_api_key=kie_api_key)
            if url:
                image_urls.append(url)
                print(f"🖼️  {tag} Image {idx+1} uploaded → {url[:60]}…")
            else:
                print(f"⚠️  {tag} Image upload {idx+1} failed, skipped.")

        # ── 2. Déterminer le generationType ───────────────────────
        if image_urls:
            generation_type = "FIRST_AND_LAST_FRAMES_2_VIDEO"
        else:
            generation_type = "TEXT_2_VIDEO"

        # ── 3. Soumission ─────────────────────────────────────────
        payload = {
            "prompt":            prompt,
            "model":             kie_model,
            "generationType":    generation_type,
            "aspect_ratio":      aspect_ratio,
            "resolution":        resolution,
            "enableTranslation": True,
        }
        if image_urls:
            payload["imageUrls"] = image_urls

        # Option B: session locale par batch (evite stale connections du pool global)
        # Option C: jitter 0-0.5s avant POST en batch (evite burst simultane)
        _local_kie = _make_kie_session()
        if _batch_idx is not None:
            import random as _random_jitter
            time.sleep(_random_jitter.uniform(0, 0.5))
        try:
            resp = _local_kie.post(KIE_VEO_GENERATE_URL, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 200:
                return self._handle_error(f"❌ {tag} Submission rejected: {data.get('msg')}")
            task_id = data.get("data", {}).get("taskId")
            if not task_id:
                return self._handle_error(f"❌ {tag} No taskId in response: {data}")
            print(f"🔖 {tag} Task ID : {task_id}")
        except requests.RequestException as e:
            return self._handle_error(f"❌ {tag} Submission error : {e}")

        # ── 4. Polling ────────────────────────────────────────────
        elapsed      = 0
        result_data  = {}
        _debug_logged = False
        print(f"⏳ {tag} Waiting for result (timeout: {_VIDEO_TIMEOUT_S}s)...")
        while elapsed < _VIDEO_TIMEOUT_S:
            time.sleep(_VIDEO_POLL_DELAY)
            elapsed += _VIDEO_POLL_DELAY
            try:
                poll_resp = _local_kie.get(
                    KIE_VEO_POLL_URL, params={"taskId": task_id},
                    headers=headers, timeout=15,
                )
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
            except requests.RequestException as e:
                print(f"⚠️  {tag} Polling error ({elapsed}s) : {e}")
                continue

            if poll_data.get("code") != 200:
                return self._handle_error(
                    f"❌ {tag} Poll error: {poll_data.get('msg')}"
                )
            result_data = poll_data.get("data", {})
            # Kie.ai Veo : successFlag 1 = terminé, errorCode/errorMessage = échec
            error_code = result_data.get("errorCode")
            error_msg  = result_data.get("errorMessage")
            if error_code or error_msg:
                return self._handle_error(
                    f"❌ {tag} Generation failed: [{error_code}] {error_msg}"
                )
            if result_data.get("successFlag") == 1 or result_data.get("completeTime"):
                print(f"✅ {tag} Task done! ({elapsed}s)")
                break
            print(f"   {tag} [{elapsed}s/{_VIDEO_TIMEOUT_S}s] waiting (successFlag=0)...")
        else:
            return self._handle_error(f"❌ {tag} Timeout after {_VIDEO_TIMEOUT_S}s.")

        # ── 5. Récupération de l'URL vidéo ────────────────────────
        try:
            import json as _json
            video_url = None
            # Cas 1 : champ "response" = URL directe (Kie.ai Veo)
            raw_response = result_data.get("response")
            if isinstance(raw_response, str) and raw_response.startswith("http"):
                video_url = raw_response
            # Cas 2 : "response" est un JSON sérialisé
            elif isinstance(raw_response, str):
                try:
                    parsed = _json.loads(raw_response)
                    video_url = (
                        parsed.get("videoUrl")
                        or parsed.get("video_url")
                        or (parsed.get("resultUrls") or [None])[0]
                    )
                except Exception:
                    pass
            # Cas 3 : "response" est déjà un dict
            elif isinstance(raw_response, dict):
                video_url = (
                    raw_response.get("videoUrl")
                    or raw_response.get("video_url")
                    or (raw_response.get("resultUrls") or [None])[0]
                )
            # Cas 4 : videoUrl ou resultJson au niveau de data (fallback)
            if not video_url:
                video_url = result_data.get("videoUrl") or result_data.get("video_url")
            if not video_url:
                result_json = _json.loads(result_data.get("resultJson", "{}"))
                video_url = (result_json.get("resultUrls") or [None])[0]
            if not video_url:
                return self._handle_error(
                    f"❌ {tag} No video URL in response.\ndata={result_data}"
                )
            print(f"🎬 {tag} Video URL: {video_url[:80]}…")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Response parsing : {e}")

        # ── 6. Download de la vidéo ─────────────────────────
        try:
            dl_resp = requests.get(video_url, timeout=120)
            dl_resp.raise_for_status()
            video_bytes = dl_resp.content
            print(f"✅ {tag} Video downloaded ({len(video_bytes)} bytes).")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Video download error : {e}")

        return self._finalize_video(video_bytes, tag, kie_model, aspect_ratio, duration)

    # ─────────────────────────────────────────────────────────────
    #  VIDEO — Kling 3.0 Motion Control — Kie.ai
    # ─────────────────────────────────────────────────────────────
    def _generate_kling_motion_control_kie(
        self,
        prompt          = "",
        resolution      = "720p",
        kie_api_key     = "",
        ws_api_key      = "",
        image_tensors   = None,
        video_reference = None,
        _batch_idx     = None,
    ):
        mode    = "1080p" if resolution == "1080p" else "720p"
        tag     = f"[Kie.ai Kling 3.0 Motion Control {mode}]"
        headers = {
            "Authorization": f"Bearer {kie_api_key}",
            "Content-Type":  "application/json",
        }

        if not video_reference:
            return self._handle_error(
                f"❌ {tag} video_reference missing.\n"
                "→ Connect a reference video to the 'video_reference' input."
            )

        print(f"🎬 {tag} Initialisation — mode={mode}")

        # ── 1. Upload image de référence (optionnel) ──────────────
        input_urls = []
        if image_tensors:
            url = self._tensor_to_public_url(image_tensors[0], idx=1, ws_api_key=ws_api_key, kie_api_key=kie_api_key)
            if url:
                input_urls.append(url)
                print(f"🖼️  {tag} Image uploaded → {url[:60]}…")
            else:
                print(f"⚠️  {tag} Image upload failed — will be skipped.")

        # ── 2. Upload vidéo de référence ──────────────────────────
        print(f"🎞️  {tag} Uploading reference video...")
        video_url = self._video_to_public_url(video_reference, ws_api_key=ws_api_key, kie_api_key=kie_api_key)
        if not video_url:
            return self._handle_error(
                f"❌ {tag} Unable to upload reference video."
            )
        print(f"🎞️  {tag} Reference video → {video_url[:60]}…")

        # ── 3. Soumission ─────────────────────────────────────────
        payload = {
            "model": "kling-3.0/motion-control",
            "input": {
                "prompt":                prompt,
                "input_urls":            input_urls,
                "video_urls":            [video_url],
                "mode":                  mode,
                "character_orientation": "video",
                "background_source":     "input_video",
            },
        }
        print(f"🚀 {tag} Submitting...")

        # Option B: session locale par batch (evite stale connections du pool global)
        # Option C: jitter 0-0.5s avant POST en batch (evite burst simultane)
        _local_kie = _make_kie_session()
        if _batch_idx is not None:
            import random as _random_jitter
            time.sleep(_random_jitter.uniform(0, 0.5))
        try:
            resp = _local_kie.post(KIE_CREATE_URL, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            return self._handle_error(f"❌ {tag} Submission error : {e}")

        if data.get("code") != 200:
            return self._handle_error(
                f"❌ {tag} Submission rejected: {data.get('msg', 'unknown error')}"
            )
        task_id = data.get("data", {}).get("taskId")
        if not task_id:
            return self._handle_error(f"❌ {tag} No taskId: {data}")
        print(f"🔖 {tag} Task ID : {task_id}")

        # ── 4. Polling ────────────────────────────────────────────
        elapsed     = 0
        result_data = {}
        print(f"⏳ {tag} Waiting for result (timeout: {_MOTION_CONTROL_TIMEOUT_S}s)...")
        while elapsed < _MOTION_CONTROL_TIMEOUT_S:
            time.sleep(_VIDEO_POLL_DELAY)
            elapsed += _VIDEO_POLL_DELAY
            try:
                poll_resp = _local_kie.get(
                    KIE_POLL_URL, params={"taskId": task_id},
                    headers=headers, timeout=15,
                )
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
            except requests.RequestException as e:
                print(f"⚠️  {tag} Polling error ({elapsed}s) : {e}")
                continue

            if poll_data.get("code") != 200:
                return self._handle_error(
                    f"❌ {tag} Poll error: {poll_data.get('msg')}"
                )
            result_data = poll_data.get("data", {})
            state       = result_data.get("state", "")
            if state == "success":
                print(f"✅ {tag} Task done! ({elapsed}s)")
                break
            elif state in ("fail", "failed", "error"):
                return self._handle_error(
                    f"❌ {tag} Failed: {result_data.get('failMsg', '?')}"
                )
            else:
                print(f"   {tag} [{elapsed}s/{_MOTION_CONTROL_TIMEOUT_S}s] state={state!r}...")
        else:
            return self._handle_error(f"❌ {tag} Timeout after {_MOTION_CONTROL_TIMEOUT_S}s.")

        # ── 5. Récupération URL vidéo ─────────────────────────────
        try:
            import json as _json
            video_out = None
            video_out = result_data.get("videoUrl") or result_data.get("video_url")
            if not video_out:
                result_json = _json.loads(result_data.get("resultJson", "{}"))
                video_out   = (result_json.get("resultUrls") or [None])[0]
            if not video_out:
                raw = result_data.get("response")
                if isinstance(raw, str) and raw.startswith("http"):
                    video_out = raw
                elif isinstance(raw, str):
                    try:
                        parsed    = _json.loads(raw)
                        video_out = (
                            parsed.get("videoUrl")
                            or parsed.get("video_url")
                            or (parsed.get("resultUrls") or [None])[0]
                        )
                    except Exception:
                        pass
                elif isinstance(raw, dict):
                    video_out = raw.get("videoUrl") or raw.get("video_url")
            if not video_out:
                return self._handle_error(
                    f"❌ {tag} No video URL in response.\ndata={result_data}"
                )
            print(f"🎬 {tag} URL : {video_out[:80]}…")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Response parsing : {e}")

        # ── 6. Download ─────────────────────────────────────
        try:
            dl_resp = requests.get(video_out, timeout=120)
            dl_resp.raise_for_status()
            video_bytes = dl_resp.content
            print(f"✅ {tag} Video downloaded ({len(video_bytes)} bytes).")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Download error : {e}")

        return self._finalize_video(video_bytes, tag, "kling-3.0/motion-control", "16:9", 0)

    # ─────────────────────────────────────────────────────────────
    #  VIDEO — Kling 2.6 — Kie.ai
    # ─────────────────────────────────────────────────────────────
    def _generate_kling26_kie(
        self,
        prompt         = "",
        aspect_ratio   = "16:9",
        duration       = 5,
        generate_audio = True,
        kie_api_key    = "",
        ws_api_key     = "",
        image_tensors  = None,
        _batch_idx     = None,
    ):
        tag     = "[Kie.ai Kling 2.6]"
        headers = {
            "Authorization": f"Bearer {kie_api_key}",
            "Content-Type":  "application/json",
        }

        # ── Durée : 5 ou 10s uniquement ──────────────────────────
        dur = 10 if int(duration) > 7 else 5

        print(f"🎬 {tag} Initializing — duration={dur}s | audio={generate_audio}")

        # ── 1. Upload image de référence (optionnel) ──────────────
        image_urls = []
        for idx, tensor in enumerate(image_tensors or []):
            url = self._tensor_to_public_url(tensor, idx=idx + 1, ws_api_key=ws_api_key, kie_api_key=kie_api_key)
            if url:
                image_urls.append(url)
                print(f"🖼️  {tag} Image {idx+1} uploaded → {url[:60]}…")
            else:
                print(f"⚠️  {tag} Image upload {idx+1} failed, skipped.")
            break   # Kie.ai Kling 2.6 accepte max 1 image

        # ── 2. Soumission ─────────────────────────────────────────
        payload = {
            "model": "kling-2.6/image-to-video",
            "input": {
                "prompt":     prompt,
                "image_urls": image_urls,
                "sound":      generate_audio,
                "duration":   str(dur),
            },
        }
        print(f"🚀 {tag} Submitting...")

        # Option B: session locale par batch (evite stale connections du pool global)
        # Option C: jitter 0-0.5s avant POST en batch (evite burst simultane)
        _local_kie = _make_kie_session()
        if _batch_idx is not None:
            import random as _random_jitter
            time.sleep(_random_jitter.uniform(0, 0.5))
        try:
            resp = _local_kie.post(KIE_CREATE_URL, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            return self._handle_error(f"❌ {tag} Submission error : {e}")

        if data.get("code") != 200:
            return self._handle_error(
                f"❌ {tag} Submission rejected: {data.get('msg', 'unknown error')}"
            )
        task_id = data.get("data", {}).get("taskId")
        if not task_id:
            return self._handle_error(f"❌ {tag} No taskId: {data}")
        print(f"🔖 {tag} Task ID : {task_id}")

        # ── 3. Polling ────────────────────────────────────────────
        elapsed     = 0
        result_data = {}
        print(f"⏳ {tag} Waiting for result (timeout: {_VIDEO_TIMEOUT_S}s)...")
        while elapsed < _VIDEO_TIMEOUT_S:
            time.sleep(_VIDEO_POLL_DELAY)
            elapsed += _VIDEO_POLL_DELAY
            try:
                poll_resp = _local_kie.get(
                    KIE_POLL_URL, params={"taskId": task_id},
                    headers=headers, timeout=15,
                )
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
            except requests.RequestException as e:
                print(f"⚠️  {tag} Polling error ({elapsed}s) : {e}")
                continue

            if poll_data.get("code") != 200:
                return self._handle_error(
                    f"❌ {tag} Poll error: {poll_data.get('msg')}"
                )
            result_data = poll_data.get("data", {})
            state       = result_data.get("state", "")
            if state == "success":
                print(f"✅ {tag} Task done! ({elapsed}s)")
                break
            elif state in ("fail", "failed", "error"):
                return self._handle_error(
                    f"❌ {tag} Failed: {result_data.get('failMsg', '?')}"
                )
            else:
                print(f"   {tag} [{elapsed}s/{_VIDEO_TIMEOUT_S}s] state={state!r}...")
        else:
            return self._handle_error(f"❌ {tag} Timeout after {_VIDEO_TIMEOUT_S}s.")

        # ── 4. Récupération URL vidéo ─────────────────────────────
        try:
            import json as _json
            video_url = None
            # Cas 1 : champ direct
            video_url = result_data.get("videoUrl") or result_data.get("video_url")
            # Cas 2 : resultJson → resultUrls
            if not video_url:
                result_json = _json.loads(result_data.get("resultJson", "{}"))
                video_url   = (result_json.get("resultUrls") or [None])[0]
            # Cas 3 : response field (comme Veo)
            if not video_url:
                raw = result_data.get("response")
                if isinstance(raw, str) and raw.startswith("http"):
                    video_url = raw
                elif isinstance(raw, str):
                    try:
                        parsed    = _json.loads(raw)
                        video_url = (
                            parsed.get("videoUrl")
                            or parsed.get("video_url")
                            or (parsed.get("resultUrls") or [None])[0]
                        )
                    except Exception:
                        pass
                elif isinstance(raw, dict):
                    video_url = raw.get("videoUrl") or raw.get("video_url")
            if not video_url:
                return self._handle_error(
                    f"❌ {tag} No video URL in response.\ndata={result_data}"
                )
            print(f"🎬 {tag} URL : {video_url[:80]}…")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Response parsing : {e}")

        # ── 5. Download ─────────────────────────────────────
        try:
            dl_resp = requests.get(video_url, timeout=120)
            dl_resp.raise_for_status()
            video_bytes = dl_resp.content
            print(f"✅ {tag} Video downloaded ({len(video_bytes)} bytes).")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Download error : {e}")

        return self._finalize_video(video_bytes, tag, "kling-2.6/image-to-video", aspect_ratio, dur)

    # ─────────────────────────────────────────────────────────────
    #  VIDEO — Kling 3.0 — Kie.ai
    #   Endpoint unique : model="kling-3.0/video"
    #   Mode (std / pro) pilotés par le paramètre "mode" du payload.
    #   Jusqu'à 2 images : [0]=première frame, [1]=dernière frame.
    # ─────────────────────────────────────────────────────────────
    def _generate_kling30_kie(
        self,
        prompt         = "",
        aspect_ratio   = "16:9",
        duration       = 5,
        resolution     = "1080p",
        generate_audio = True,
        kie_api_key    = "",
        ws_api_key     = "",
        image_tensors  = None,
        _batch_idx     = None,
    ):
        # 4K = "4K" | pro = 1080p | std = 720p
        mode_api = "4K" if resolution == "4K" else ("pro" if resolution == "1080p" else "std")
        tag      = f"[Kie.ai Kling 3.0 {'Pro' if mode_api == 'pro' else 'Std'}]"
        headers  = {
            "Authorization": f"Bearer {kie_api_key}",
            "Content-Type":  "application/json",
        }

        # ── Durée : 3 → 15s, entier ──────────────────────────────
        dur = max(3, min(15, int(duration)))

        print(f"🎬 {tag} Starting — duration={dur}s | mode={mode_api} | "
              f"ratio={aspect_ratio} | audio={generate_audio}")

        # ── 1. Upload images de référence (max 2 : start + end) ──
        #    Kling 3.0 accepte une image de début (index 0) et une
        #    image de fin (index 1). Les suivantes sont ignoredes.
        image_urls = []
        tensors = list(image_tensors or [])
        if len(tensors) > 2:
            print(f"⚠️  {tag} {len(tensors)} images fournies — seules les 2 "
                  f"first (start + end) will be used.")
            tensors = tensors[:2]
        for idx, tensor in enumerate(tensors):
            url = self._tensor_to_public_url(tensor, idx=idx + 1, ws_api_key=ws_api_key, kie_api_key=kie_api_key)
            if url:
                image_urls.append(url)
                role = "start" if idx == 0 else "end"
                print(f"🖼️  {tag} Image {role} ({idx+1}/{len(tensors)}) uploaded → {url[:60]}…")
            else:
                print(f"⚠️  {tag} Image upload {idx+1} failed, skipped.")

        # ── 2. Payload ───────────────────────────────────────────
        payload = {
            "model": "kling-3.0/video",
            "input": {
                "prompt":       prompt,
                "image_urls":   image_urls,
                "sound":        bool(generate_audio),
                "duration":     str(dur),
                "aspect_ratio": aspect_ratio,
                "mode":         mode_api,
                "multi_shots":  False,
            },
        }
        print(f"🚀 {tag} Submitting...")

        # Option B: session locale par batch (evite stale connections du pool global)
        # Option C: jitter 0-0.5s avant POST en batch (evite burst simultane)
        _local_kie = _make_kie_session()
        if _batch_idx is not None:
            import random as _random_jitter
            time.sleep(_random_jitter.uniform(0, 0.5))
        try:
            resp = _local_kie.post(KIE_CREATE_URL, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            return self._handle_error(f"❌ {tag} Submission error : {e}")

        if data.get("code") != 200:
            return self._handle_error(
                f"❌ {tag} Submission rejected: {data.get('msg', 'unknown error')}"
            )
        task_id = data.get("data", {}).get("taskId")
        if not task_id:
            return self._handle_error(f"❌ {tag} No taskId: {data}")
        print(f"🔖 {tag} Task ID : {task_id}")

        # ── 3. Polling ───────────────────────────────────────────
        elapsed     = 0
        result_data = {}
        print(f"⏳ {tag} Waiting for result (timeout: {_VIDEO_TIMEOUT_S}s)...")
        while elapsed < _VIDEO_TIMEOUT_S:
            time.sleep(_VIDEO_POLL_DELAY)
            elapsed += _VIDEO_POLL_DELAY
            try:
                poll_resp = _local_kie.get(
                    KIE_POLL_URL, params={"taskId": task_id},
                    headers=headers, timeout=15,
                )
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
            except requests.RequestException as e:
                print(f"⚠️  {tag} Polling error ({elapsed}s) : {e}")
                continue

            if poll_data.get("code") != 200:
                return self._handle_error(
                    f"❌ {tag} Poll error: {poll_data.get('msg')}"
                )
            result_data = poll_data.get("data", {})
            state       = result_data.get("state", "")
            if state == "success":
                print(f"✅ {tag} Task done! ({elapsed}s)")
                break
            elif state in ("fail", "failed", "error"):
                return self._handle_error(
                    f"❌ {tag} Failed: {result_data.get('failMsg', '?')}"
                )
            else:
                print(f"   {tag} [{elapsed}s/{_VIDEO_TIMEOUT_S}s] state={state!r}...")
        else:
            return self._handle_error(f"❌ {tag} Timeout after {_VIDEO_TIMEOUT_S}s.")

        # ── 4. Récupération URL vidéo ────────────────────────────
        try:
            import json as _json
            video_url = result_data.get("videoUrl") or result_data.get("video_url")
            if not video_url:
                result_json = _json.loads(result_data.get("resultJson", "{}"))
                video_url   = (result_json.get("resultUrls") or [None])[0]
            if not video_url:
                raw = result_data.get("response")
                if isinstance(raw, str) and raw.startswith("http"):
                    video_url = raw
                elif isinstance(raw, str):
                    try:
                        parsed    = _json.loads(raw)
                        video_url = (
                            parsed.get("videoUrl")
                            or parsed.get("video_url")
                            or (parsed.get("resultUrls") or [None])[0]
                        )
                    except Exception:
                        pass
                elif isinstance(raw, dict):
                    video_url = raw.get("videoUrl") or raw.get("video_url")
            if not video_url:
                return self._handle_error(
                    f"❌ {tag} No video URL in response.\ndata={result_data}"
                )
            print(f"🎬 {tag} URL : {video_url[:80]}…")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Response parsing : {e}")

        # ── 5. Download ────────────────────────────────────
        try:
            dl_resp = requests.get(video_url, timeout=120)
            dl_resp.raise_for_status()
            video_bytes = dl_resp.content
            print(f"✅ {tag} Video downloaded ({len(video_bytes)} bytes).")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Download error : {e}")

        return self._finalize_video(video_bytes, tag, "kling-3.0/video", aspect_ratio, dur)

    # ─────────────────────────────────────────────────────────────
    #  VIDEO — Seedance 2.0 — WaveSpeed
    # ─────────────────────────────────────────────────────────────
    def _generate_seedance_wavespeed(
        self,
        prompt         = "",
        aspect_ratio   = "16:9",
        duration       = 5,
        resolution     = "720p",
        generate_audio = True,
        ws_api_key     = "",
        image_tensors  = None,
    ):
        url     = f"{WAVESPEED_BASE_URL}/{WS_SEEDANCE20_SLUG}"
        tag     = f"[WaveSpeed Seedance 2.0 {resolution}]"
        headers = {
            "Authorization": f"Bearer {ws_api_key}",
            "Content-Type":  "application/json",
        }

        # Durée : 4–15s
        dur = max(4, min(15, int(duration)))
        print(f"🎬 {tag} Starting — duration={dur}s | resolution={resolution} | ratio={aspect_ratio}")

        # ── 1. Upload image de début (obligatoire) ────────────────
        tensors = list(image_tensors or [])
        image_url     = None
        last_image_url = None

        if tensors:
            image_url = self._tensor_to_public_url(tensors[0], idx=1, ws_api_key=ws_api_key)
            if image_url:
                print(f"🖼️  {tag} Start image uploaded → {image_url[:60]}…")
            else:
                print(f"⚠️  {tag} Image upload start failed.")
        else:
            print(f"⚠️  {tag} Seedance 2.0 est image-to-video — aucune image fournie.")

        if len(tensors) >= 2:
            last_image_url = self._tensor_to_public_url(tensors[1], idx=2, ws_api_key=ws_api_key)
            if last_image_url:
                print(f"🖼️  {tag} End image uploaded → {last_image_url[:60]}…")
            else:
                print(f"⚠️  {tag} Image upload end failed, skipped.")

        # ── 2. Payload ────────────────────────────────────────────
        payload: dict = {
            "prompt":     prompt,
            "duration":   dur,
            "resolution": resolution,
        }
        # Aspect ratio (facultatif — s'adapte à l'image si non spécifié)
        if aspect_ratio in ("16:9", "9:16", "4:3", "3:4", "1:1", "21:9"):
            payload["aspect_ratio"] = aspect_ratio
        if image_url:
            payload["image"] = image_url
        if last_image_url:
            payload["last_image"] = last_image_url

        # ── 3. Soumission ─────────────────────────────────────────
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data    = resp.json()
            task_id = (data.get("data") or {}).get("id") or data.get("id")
            if not task_id:
                return self._handle_error(f"❌ {tag} Submission failed: {data}")
            print(f"🔖 {tag} Task ID : {task_id}")
        except requests.RequestException as e:
            return self._handle_error(f"❌ {tag} Submission error : {e}")

        # ── 4. Polling ────────────────────────────────────────────
        poll_url  = WAVESPEED_POLL_URL.format(task_id=task_id)
        elapsed   = 0
        poll_data = {}
        print(f"⏳ {tag} Waiting for result (timeout: {_VIDEO_TIMEOUT_S}s)...")
        while elapsed < _VIDEO_TIMEOUT_S:
            time.sleep(_VIDEO_POLL_DELAY)
            elapsed += _VIDEO_POLL_DELAY
            try:
                poll_resp = requests.get(
                    poll_url,
                    headers={"Authorization": f"Bearer {ws_api_key}"},
                    timeout=15,
                )
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
            except requests.RequestException as e:
                print(f"⚠️  {tag} Polling error ({elapsed}s) : {e}")
                continue

            status = poll_data.get("data", {}).get("status", "")
            if status == "completed":
                outputs = poll_data.get("data", {}).get("outputs", [])
                if not outputs:
                    return self._handle_error(f"❌ {tag} Completed but no video output.")
                video_url = outputs[0]
                print(f"✅ {tag} Video ready! ({elapsed}s)")
                break
            elif status == "failed":
                error_msg = poll_data.get("data", {}).get("error", "Error inconnue")
                return self._handle_error(f"❌ {tag} Generation failed: {error_msg}")
            else:
                print(f"   {tag} [{elapsed}s/{_VIDEO_TIMEOUT_S}s] status={status!r}...")
        else:
            try:
                requests.delete(
                    WAVESPEED_CANCEL_URL.format(task_id=task_id),
                    headers={"Authorization": f"Bearer {ws_api_key}"},
                    timeout=10,
                )
            except Exception:
                pass
            return self._handle_error(f"❌ {tag} Timeout after {_VIDEO_TIMEOUT_S}s.")

        # ── 5. Download ─────────────────────────────────────
        try:
            dl_resp = requests.get(video_url, timeout=120)
            dl_resp.raise_for_status()
            video_bytes = dl_resp.content
            print(f"✅ {tag} Video downloaded ({len(video_bytes)} bytes).")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Download error : {e}")

        return self._finalize_video(video_bytes, tag, WS_SEEDANCE20_SLUG, aspect_ratio, dur)

    # ─────────────────────────────────────────────────────────────
    #  VIDEO — Seedance 2.0 — Fal.ai
    # ─────────────────────────────────────────────────────────────
    def _generate_seedance_fal(
        self,
        prompt         = "",
        aspect_ratio   = "16:9",
        duration       = 5,
        resolution     = "720p",
        generate_audio = True,
        fal_api_key    = "",
        image_tensors  = None,
    ):
        if not _ensure_fal_client():
            return self._handle_error("❌ Unable to install fal-client.")
        import fal_client
        import time as _time
        os.environ["FAL_KEY"] = fal_api_key

        tag      = f"[Fal.ai Seedance 2.0 {resolution}]"
        endpoint = FAL_SEEDANCE20_ENDPOINT

        # Durée : "4"–"15" (string) ou "auto"
        dur = max(4, min(15, int(duration)))
        fal_duration = str(dur)

        # Aspect ratio : snap si hors liste FAL
        _SEEDANCE_FAL_AR = {"auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}
        _SEEDANCE_FAL_AR_FALLBACK = {
            "2:3": "3:4", "3:2": "4:3", "4:5": "3:4", "5:4": "4:3",
        }
        if aspect_ratio not in _SEEDANCE_FAL_AR:
            fallback = _SEEDANCE_FAL_AR_FALLBACK.get(aspect_ratio, "16:9")
            print(f"⚠️  {tag} Aspect ratio {aspect_ratio!r} not supported → fallback {fallback!r}")
            aspect_ratio = fallback

        print(f"🎬 {tag} Starting — duration={dur}s | resolution={resolution} | ratio={aspect_ratio} | audio={generate_audio}")

        tensors = list(image_tensors or [])

        # ── Upload image de début (optionnel) ─────────────────────
        image_url = None
        if tensors:
            try:
                image_url = _tensor_to_fal_url(tensors[0], fal_client, tag=tag, idx=1)
                print(f"🖼️  {tag} Start image uploaded")
            except Exception as e:
                print(f"⚠️  {tag} Image upload start failed: {e}")

        # ── Upload image de fin (optionnel) ───────────────────────
        end_image_url = None
        if len(tensors) >= 2:
            try:
                end_image_url = _tensor_to_fal_url(tensors[1], fal_client, tag=tag, idx=2)
                print(f"🖼️  {tag} End image uploaded")
            except Exception as e:
                print(f"⚠️  {tag} Image upload end failed: {e}")

        # ── Payload ───────────────────────────────────────────────
        arguments = {
            "prompt":         prompt,
            "resolution":     resolution,
            "duration":       fal_duration,
            "aspect_ratio":   aspect_ratio,
            "generate_audio": generate_audio,
        }
        if image_url:
            arguments["image_url"] = image_url
        if end_image_url:
            arguments["end_image_url"] = end_image_url

        # ── Subscribe ─────────────────────────────────────────────
        _t0 = _time.time()
        _last_log_t = [0.0]

        def _on_update(update):
            elapsed_now = int(_time.time() - _t0)
            now = _time.time()
            if isinstance(update, fal_client.InProgress):
                for log in (update.logs or []):
                    msg = log.get("message", "") if isinstance(log, dict) else str(log)
                    if msg and len(msg) <= 300 and now - _last_log_t[0] >= 10:
                        print(f"   {tag} [LOG] {msg}")
                        _last_log_t[0] = now
            elif isinstance(update, fal_client.Queued):
                if now - _last_log_t[0] >= 10:
                    pos = getattr(update, "position", "?")
                    print(f"   {tag} [{elapsed_now}s] Queue position={pos}")
                    _last_log_t[0] = now
            else:
                if now - _last_log_t[0] >= 10:
                    print(f"   {tag} [{elapsed_now}s] En cours...")
                    _last_log_t[0] = now

        print(f"🚀 {tag} Submitting...")
        try:
            result_data = fal_client.subscribe(
                endpoint,
                arguments=arguments,
                with_logs=True,
                on_queue_update=_on_update,
            )
            elapsed = int(_time.time() - _t0)
            print(f"✅ {tag} Done ({elapsed}s)")
        except Exception as e:
            return self._handle_error(
                f"❌ {tag} Error: {type(e).__name__} : {_fal_short_error(e)}"
            )

        # ── Download ────────────────────────────────────────
        try:
            video_info = result_data.get("video", {})
            video_url  = video_info.get("url", "") if isinstance(video_info, dict) else ""
            if not video_url:
                return self._handle_error(f"❌ {tag} No video URL in response : {result_data}")
            print(f"🎬 {tag} URL : {video_url[:80]}…")
            dl_resp = requests.get(video_url, timeout=120)
            dl_resp.raise_for_status()
            video_bytes = dl_resp.content
            print(f"✅ {tag} Video downloaded ({len(video_bytes)} bytes).")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Download error : {e}")

        return self._finalize_video(video_bytes, tag, endpoint, aspect_ratio, dur)

    # ─────────────────────────────────────────────────────────────
    #  VIDEO — Seedance 2.0 — Kie.ai
    # ─────────────────────────────────────────────────────────────
    def _generate_seedance_kie(
        self,
        prompt         = "",
        aspect_ratio   = "16:9",
        duration       = 5,
        resolution     = "720p",
        generate_audio = True,
        kie_api_key    = "",
        ws_api_key     = "",
        image_tensors  = None,
        disable_safety = False,
        _batch_idx     = None,
    ):
        tag     = f"[Kie.ai Seedance 2.0 {resolution}]"
        headers = {
            "Authorization": f"Bearer {kie_api_key}",
            "Content-Type":  "application/json",
        }

        # Durée : 4–15s
        dur = max(4, min(15, int(duration)))
        print(f"🎬 {tag} Starting — duration={dur}s | resolution={resolution} | ratio={aspect_ratio} | audio={generate_audio}")

        tensors = list(image_tensors or [])

        # ── 1. Upload images de référence ─────────────────────────
        first_frame_url = None
        last_frame_url  = None

        if tensors:
            url = self._tensor_to_public_url(tensors[0], idx=1, ws_api_key=ws_api_key, kie_api_key=kie_api_key)
            if url:
                first_frame_url = url
                print(f"🖼️  {tag} Start image uploaded → {url[:60]}…")
            else:
                print(f"⚠️  {tag} Image upload start failed.")

        if len(tensors) >= 2:
            url = self._tensor_to_public_url(tensors[1], idx=2, ws_api_key=ws_api_key, kie_api_key=kie_api_key)
            if url:
                last_frame_url = url
                print(f"🖼️  {tag} End image uploaded → {url[:60]}…")
            else:
                print(f"⚠️  {tag} Image upload end failed, skipped.")

        # ── 2. Payload ────────────────────────────────────────────
        input_params = {
            "prompt":         prompt,
            "generate_audio": bool(generate_audio),
            "resolution":     resolution,
            "aspect_ratio":   aspect_ratio,
            "duration":       int(dur),
            "nsfw_checker":   not disable_safety,
        }
        if first_frame_url:
            input_params["first_frame_url"] = first_frame_url
        if last_frame_url:
            input_params["last_frame_url"] = last_frame_url

        payload = {
            "model": KIE_SEEDANCE20_MODEL,
            "input": input_params,
        }
        print(f"🚀 {tag} Submitting...")

        # Option B: session locale par batch (evite stale connections du pool global)
        # Option C: jitter 0-0.5s avant POST en batch (evite burst simultane)
        _local_kie = _make_kie_session()
        if _batch_idx is not None:
            import random as _random_jitter
            time.sleep(_random_jitter.uniform(0, 0.5))
        try:
            resp = _local_kie.post(KIE_CREATE_URL, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            return self._handle_error(f"❌ {tag} Submission error : {e}")

        if data.get("code") != 200:
            return self._handle_error(
                f"❌ {tag} Submission rejected: {data.get('msg', 'unknown error')}"
            )
        task_id = data.get("data", {}).get("taskId")
        if not task_id:
            return self._handle_error(f"❌ {tag} No taskId: {data}")
        print(f"🔖 {tag} Task ID : {task_id}")

        # ── 3. Polling ────────────────────────────────────────────
        elapsed     = 0
        result_data = {}
        print(f"⏳ {tag} Waiting for result (timeout: {KIE_SEEDANCE_TIMEOUT_S}s)...")
        while elapsed < KIE_SEEDANCE_TIMEOUT_S:
            time.sleep(_VIDEO_POLL_DELAY)
            elapsed += _VIDEO_POLL_DELAY
            try:
                poll_resp = _local_kie.get(
                    KIE_POLL_URL, params={"taskId": task_id},
                    headers=headers, timeout=15,
                )
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()
            except requests.RequestException as e:
                print(f"⚠️  {tag} Polling error ({elapsed}s) : {e}")
                continue

            if poll_data.get("code") != 200:
                return self._handle_error(
                    f"❌ {tag} Poll error: {poll_data.get('msg')}"
                )
            result_data = poll_data.get("data", {})
            state       = result_data.get("state", "")
            if state == "success":
                print(f"✅ {tag} Task done! ({elapsed}s)")
                break
            elif state in ("fail", "failed", "error"):
                return self._handle_error(
                    f"❌ {tag} Failed: {result_data.get('failMsg', '?')}"
                )
            else:
                print(f"   {tag} [{elapsed}s/{KIE_SEEDANCE_TIMEOUT_S}s] state={state!r}...")
        else:
            return self._handle_error(f"❌ {tag} Timeout after {KIE_SEEDANCE_TIMEOUT_S}s.")

        # ── 4. Récupération URL vidéo ─────────────────────────────
        try:
            import json as _json
            video_url = result_data.get("videoUrl") or result_data.get("video_url")
            if not video_url:
                result_json = _json.loads(result_data.get("resultJson", "{}"))
                video_url   = (result_json.get("resultUrls") or [None])[0]
            if not video_url:
                raw = result_data.get("response")
                if isinstance(raw, str) and raw.startswith("http"):
                    video_url = raw
                elif isinstance(raw, str):
                    try:
                        parsed    = _json.loads(raw)
                        video_url = (
                            parsed.get("videoUrl")
                            or parsed.get("video_url")
                            or (parsed.get("resultUrls") or [None])[0]
                        )
                    except Exception:
                        pass
                elif isinstance(raw, dict):
                    video_url = raw.get("videoUrl") or raw.get("video_url")
            if not video_url:
                return self._handle_error(
                    f"❌ {tag} No video URL in response.\ndata={result_data}"
                )
            print(f"🎬 {tag} URL : {video_url[:80]}…")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Response parsing : {e}")

        # ── 5. Download ─────────────────────────────────────
        try:
            dl_resp = requests.get(video_url, timeout=120)
            dl_resp.raise_for_status()
            video_bytes = dl_resp.content
            print(f"✅ {tag} Video downloaded ({len(video_bytes)} bytes).")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Download error : {e}")

        return self._finalize_video(video_bytes, tag, KIE_SEEDANCE20_MODEL, aspect_ratio, dur)

    # ─────────────────────────────────────────────────────────────
    #  VIDEO — Fal.ai (Veo 3.1 / Fast / Lite)
    # ─────────────────────────────────────────────────────────────
    #  VIDEO — Kling 2.6 Pro — Fal.ai
    # ─────────────────────────────────────────────────────────────
    def _generate_kling26_fal(
        self,
        prompt         = "",
        aspect_ratio   = "16:9",
        duration       = 5,
        resolution     = "1080p",
        generate_audio = True,
        fal_api_key    = "",
        image_tensors  = None,
    ):
        if not _ensure_fal_client():
            return self._handle_error("❌ Unable to install fal-client.")
        import fal_client
        import time as _time
        os.environ["FAL_KEY"] = fal_api_key

        tag = "[Fal.ai Kling 2.6 Pro]"

        # ── Résolution : FAL n'a que Pro (1080p) ──────────────────
        if resolution != "1080p":
            print(f"⚠️  {tag} Resolution {resolution!r} not available on FAL — "
                  f"utilisation de 1080p (Pro uniquement).")

        # ── Durée : 5 ou 10s uniquement ───────────────────────────
        dur = 10 if int(duration) > 7 else 5
        fal_duration = str(dur)   # API attend "5" ou "10"

        # ── Aspect ratio : snap si hors liste ─────────────────────
        if aspect_ratio not in _KLING26_FAL_AR:
            fallback = _KLING26_FAL_AR_FALLBACK.get(aspect_ratio, "16:9")
            print(f"⚠️  {tag} Aspect ratio {aspect_ratio!r} not supported → fallback {fallback!r}")
            aspect_ratio = fallback

        has_image = bool(image_tensors)
        endpoint  = FAL_KLING26_I2V_ENDPOINT if has_image else FAL_KLING26_T2V_ENDPOINT
        mode      = "img2vid" if has_image else "txt2vid"

        print(f"🎬 {tag} [{mode}] ratio={aspect_ratio} | duration={dur}s | audio={generate_audio}")

        # ── Upload image (optionnel) ──────────────────────────────
        image_url = None
        if has_image:
            try:
                image_url = _tensor_to_fal_url(image_tensors[0], fal_client, tag=tag, idx=1)
                print(f"🖼️  {tag} Image uploaded")
            except Exception as e:
                print(f"⚠️  {tag} Image upload failed: {e} — text-to-video mode.")
                endpoint = FAL_KLING26_T2V_ENDPOINT
                mode     = "txt2vid"

        # ── Payload ───────────────────────────────────────────────
        arguments = {
            "prompt":           prompt,
            "duration":         fal_duration,
            "aspect_ratio":     aspect_ratio,
            "negative_prompt":  "blur, distort, and low quality",
            "generate_audio":   generate_audio,
        }
        if image_url:
            arguments["start_image_url"] = image_url

        # ── Subscribe ─────────────────────────────────────────────
        _t0 = _time.time()
        _last_log_t = [0.0]

        def _on_update(update):
            elapsed_now = int(_time.time() - _t0)
            now = _time.time()
            if isinstance(update, fal_client.InProgress):
                for log in (update.logs or []):
                    msg = log.get("message", "") if isinstance(log, dict) else str(log)
                    if msg and len(msg) <= 300 and now - _last_log_t[0] >= 10:
                        print(f"   {tag} [LOG] {msg}")
                        _last_log_t[0] = now
            elif isinstance(update, fal_client.Queued):
                if now - _last_log_t[0] >= 10:
                    pos = getattr(update, "position", "?")
                    print(f"   {tag} [{elapsed_now}s] Queue position={pos}")
                    _last_log_t[0] = now
            else:
                if now - _last_log_t[0] >= 10:
                    print(f"   {tag} [{elapsed_now}s] En cours...")
                    _last_log_t[0] = now

        try:
            result_data = fal_client.subscribe(
                endpoint,
                arguments=arguments,
                with_logs=True,
                on_queue_update=_on_update,
            )
            elapsed = int(_time.time() - _t0)
            print(f"✅ {tag} Done ({elapsed}s)")
        except Exception as e:
            return self._handle_error(
                f"❌ {tag} Error: {type(e).__name__} : {_fal_short_error(e)}"
            )

        # ── Download ────────────────────────────────────────
        try:
            video_info = result_data.get("video", {})
            video_url  = video_info.get("url", "") if isinstance(video_info, dict) else ""
            if not video_url:
                return self._handle_error(f"❌ {tag} No video URL in response : {result_data}")
            print(f"🎬 {tag} URL : {video_url[:80]}…")
            dl_resp = requests.get(video_url, timeout=120)
            dl_resp.raise_for_status()
            video_bytes = dl_resp.content
            print(f"✅ {tag} Video downloaded ({len(video_bytes)} bytes).")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Download error : {e}")

        return self._finalize_video(video_bytes, tag, endpoint, aspect_ratio, dur)

    # ─────────────────────────────────────────────────────────────
    #  VIDEO — Kling 3.0 (Standard 720p / Pro 1080p) — Fal.ai
    # ─────────────────────────────────────────────────────────────
    def _generate_kling30_fal(
        self,
        prompt         = "",
        aspect_ratio   = "16:9",
        duration       = 5,
        resolution     = "720p",
        generate_audio = True,
        fal_api_key    = "",
        image_tensors  = None,
    ):
        if not _ensure_fal_client():
            return self._handle_error("❌ Unable to install fal-client.")
        import fal_client
        import time as _time
        os.environ["FAL_KEY"] = fal_api_key

        # ── Tag et endpoint selon resolution ──────────────────────
        if resolution == "4K":
            tag = "[Fal.ai Kling 3.0 4K]"
        elif resolution == "1080p":
            tag = "[Fal.ai Kling 3.0 Pro]"
        else:
            tag = "[Fal.ai Kling 3.0 Standard]"

        # ── Durée : 3–15s, chaîne ─────────────────────────────────
        dur = max(_VIDEO_DURATION_MIN, min(_VIDEO_DURATION_MAX, int(duration)))
        fal_duration = str(dur)

        # ── Aspect ratio : snap si hors liste ─────────────────────
        if aspect_ratio not in _KLING30_FAL_AR:
            fallback = _KLING30_FAL_AR_FALLBACK.get(aspect_ratio, "16:9")
            print(f"⚠️  {tag} Aspect ratio {aspect_ratio!r} not supported → fallback {fallback!r}")
            aspect_ratio = fallback

        has_image = bool(image_tensors)
        if resolution == "4K":
            endpoint = FAL_KLING30_4K_I2V_ENDPOINT if has_image else FAL_KLING30_4K_T2V_ENDPOINT
        elif resolution == "1080p":
            endpoint = FAL_KLING30_PRO_I2V_ENDPOINT if has_image else FAL_KLING30_PRO_T2V_ENDPOINT
        else:
            endpoint = FAL_KLING30_STD_I2V_ENDPOINT if has_image else FAL_KLING30_STD_T2V_ENDPOINT
        mode = "img2vid" if has_image else "txt2vid"

        print(f"🎬 {tag} [{mode}] ratio={aspect_ratio} | duration={dur}s | audio={generate_audio}")

        # ── Upload image (optionnel) ──────────────────────────────
        image_url = None
        if has_image:
            try:
                image_url = _tensor_to_fal_url(image_tensors[0], fal_client, tag=tag, idx=1)
                print(f"🖼️  {tag} Image uploaded")
            except Exception as e:
                print(f"⚠️  {tag} Image upload failed: {e} — text-to-video mode.")
                if resolution == "4K":
                    endpoint = FAL_KLING30_4K_T2V_ENDPOINT
                elif resolution == "1080p":
                    endpoint = FAL_KLING30_PRO_T2V_ENDPOINT
                else:
                    endpoint = FAL_KLING30_STD_T2V_ENDPOINT
                mode = "txt2vid"

        # ── Payload ───────────────────────────────────────────────
        arguments = {
            "prompt":           prompt,
            "duration":         fal_duration,
            "aspect_ratio":     aspect_ratio,
            "negative_prompt":  "blur, distort, and low quality",
            "generate_audio":   generate_audio,
            "cfg_scale":        0.5,
        }
        if image_url:
            arguments["start_image_url"] = image_url

        # ── Subscribe ─────────────────────────────────────────────
        _t0 = _time.time()
        _last_log_t = [0.0]

        def _on_update(update):
            elapsed_now = int(_time.time() - _t0)
            now = _time.time()
            if isinstance(update, fal_client.InProgress):
                for log in (update.logs or []):
                    msg = log.get("message", "") if isinstance(log, dict) else str(log)
                    if msg and len(msg) <= 300 and now - _last_log_t[0] >= 10:
                        print(f"   {tag} [LOG] {msg}")
                        _last_log_t[0] = now
            elif isinstance(update, fal_client.Queued):
                if now - _last_log_t[0] >= 10:
                    pos = getattr(update, "position", "?")
                    print(f"   {tag} [{elapsed_now}s] Queue position={pos}")
                    _last_log_t[0] = now
            else:
                if now - _last_log_t[0] >= 10:
                    print(f"   {tag} [{elapsed_now}s] En cours...")
                    _last_log_t[0] = now

        try:
            result_data = fal_client.subscribe(
                endpoint,
                arguments=arguments,
                with_logs=True,
                on_queue_update=_on_update,
            )
            elapsed = int(_time.time() - _t0)
            print(f"✅ {tag} Done ({elapsed}s)")
        except Exception as e:
            return self._handle_error(
                f"❌ {tag} Error: {type(e).__name__} : {_fal_short_error(e)}"
            )

        # ── Download ────────────────────────────────────────
        try:
            video_info = result_data.get("video", {})
            video_url  = video_info.get("url", "") if isinstance(video_info, dict) else ""
            if not video_url:
                return self._handle_error(f"❌ {tag} No video URL in response : {result_data}")
            print(f"🎬 {tag} URL : {video_url[:80]}…")
            dl_resp = requests.get(video_url, timeout=120)
            dl_resp.raise_for_status()
            video_bytes = dl_resp.content
            print(f"✅ {tag} Video downloaded ({len(video_bytes)} bytes).")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Download error : {e}")

        return self._finalize_video(video_bytes, tag, endpoint, aspect_ratio, dur)

    # ─────────────────────────────────────────────────────────────
    #  VIDEO — Kling 3.0 Motion Control — Fal.ai
    # ─────────────────────────────────────────────────────────────
    def _generate_kling_motion_control_fal(
        self,
        prompt          = "",
        resolution      = "720p",
        fal_api_key     = "",
        image_tensors   = None,
        video_reference = None,
    ):
        if not _ensure_fal_client():
            return self._handle_error("❌ Unable to install fal-client.")
        import fal_client
        import time as _time
        os.environ["FAL_KEY"] = fal_api_key

        is_pro = (resolution == "1080p")
        tag     = f"[Fal.ai Kling 3.0 Motion Control {'Pro' if is_pro else 'Standard'}]"
        endpoint = FAL_KLING30_PRO_MC_ENDPOINT if is_pro else FAL_KLING30_STD_MC_ENDPOINT

        if not video_reference:
            return self._handle_error(
                f"❌ {tag} video_reference missing.\n"
                "→ Connect a reference video to the 'video_reference' input."
            )

        print(f"🎬 {tag} Initialisation")

        # ── 1. Upload image de référence (optionnel) ──────────────
        image_url = None
        if image_tensors:
            try:
                image_url = _tensor_to_fal_url(image_tensors[0], fal_client, tag=tag, idx=1)
                print(f"🖼️  {tag} Image uploaded")
            except Exception as e:
                print(f"⚠️  {tag} Image upload failed: {e} — will be skipped.")

        # ── 2. Upload vidéo de référence ──────────────────────────
        print(f"🎞️  {tag} Uploading reference video...")
        try:
            with open(video_reference, "rb") as f:
                video_bytes_ref = f.read()
            ext  = os.path.splitext(video_reference)[1].lower()
            mime = "video/mp4" if ext in (".mp4", ".m4v") else "video/webm" if ext == ".webm" else "video/mp4"
            # Tentative 1 : upload vers FAL CDN
            try:
                video_url = fal_client.upload(video_bytes_ref, content_type=mime)
                print(f"🎞️  {tag} Reference video uploaded → {video_url[:60]}…")
            except Exception as upload_err:
                # Fallback base64 data URI (FAL l'accepte nativement)
                print(f"⚠️  {tag} FAL upload failed ({type(upload_err).__name__}) "
                      f"→ fallback base64 data URI ({len(video_bytes_ref)//1024} kB)")
                video_url = f"data:{mime};base64,{base64.b64encode(video_bytes_ref).decode('ascii')}"
                print(f"🎞️  {tag} Video encoded as base64.")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Reference video read error : {e}")

        # ── 3. Payload ────────────────────────────────────────────
        arguments = {
            "prompt":                prompt,
            "video_url":             video_url,
            "keep_original_sound":   True,
            "character_orientation": "video",
        }
        if image_url:
            arguments["image_url"] = image_url

        # ── 4. Subscribe ──────────────────────────────────────────
        import threading as _threading

        _t0           = _time.time()
        _stop_log     = _threading.Event()
        _LOG_INTERVAL = 30   # secondes entre chaque ligne heartbeat

        def _progress_logger():
            """Affiche un heartbeat toutes les _LOG_INTERVAL secondes."""
            while not _stop_log.wait(_LOG_INTERVAL):
                elapsed_now = int(_time.time() - _t0)
                print(f"   {tag} [{elapsed_now}s] Generating...")

        _last_log_t = [0.0]

        def _on_update(update):
            elapsed_now = int(_time.time() - _t0)
            now = _time.time()
            if isinstance(update, fal_client.InProgress):
                for log in (update.logs or []):
                    msg = log.get("message", "") if isinstance(log, dict) else str(log)
                    if msg and len(msg) <= 300 and now - _last_log_t[0] >= 10:
                        print(f"   {tag} [LOG] {msg}")
                        _last_log_t[0] = now
            elif isinstance(update, fal_client.Queued):
                if now - _last_log_t[0] >= 10:
                    pos = getattr(update, "position", "?")
                    print(f"   {tag} [{elapsed_now}s] Queue position={pos}")
                    _last_log_t[0] = now
            else:
                if now - _last_log_t[0] >= 10:
                    print(f"   {tag} [{elapsed_now}s] En cours...")
                    _last_log_t[0] = now

        _log_thread = _threading.Thread(target=_progress_logger, daemon=True)
        _log_thread.start()

        try:
            result_data = fal_client.subscribe(
                endpoint,
                arguments=arguments,
                with_logs=True,
                on_queue_update=_on_update,
            )
            elapsed = int(_time.time() - _t0)
            print(f"✅ {tag} Done ({elapsed}s)")
        except Exception as e:
            return self._handle_error(
                f"❌ {tag} Error: {type(e).__name__} : {_fal_short_error(e)}"
            )
        finally:
            _stop_log.set()
            _log_thread.join(timeout=2)

        # ── 5. Download ─────────────────────────────────────
        try:
            video_info = result_data.get("video", {})
            video_url_out = video_info.get("url", "") if isinstance(video_info, dict) else ""
            if not video_url_out:
                return self._handle_error(f"❌ {tag} No video URL in response : {result_data}")
            print(f"🎬 {tag} URL : {video_url_out[:80]}…")
            dl_resp = requests.get(video_url_out, timeout=120)
            dl_resp.raise_for_status()
            video_bytes_out = dl_resp.content
            print(f"✅ {tag} Video downloaded ({len(video_bytes_out)} bytes).")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Download error : {e}")

        return self._finalize_video(video_bytes_out, tag, endpoint, "16:9", 0)

    # ─────────────────────────────────────────────────────────────
    def _generate_veo_fal(
        self,
        prompt           = "",
        model            = "Veo 3.1 Lite",
        aspect_ratio     = "16:9",
        duration         = 8,
        resolution       = "1080p",
        generate_audio   = True,
        fal_api_key      = "",
        safety_tolerance = "4",
        image_tensors    = None,
    ):
        if not _ensure_fal_client():
            return self._handle_error("❌ Unable to install fal-client.")
        import fal_client
        os.environ["FAL_KEY"] = fal_api_key

        endpoint = _VIDEO_MODEL_MAP_FAL[model]
        tag      = f"[Fal.ai {model}]"

        # Duration : FAL accepte "4s", "6s", "8s" uniquement
        dur_int = max(4, min(8, int(duration)))
        if dur_int <= 4:
            fal_duration = "4s"
        elif dur_int <= 6:
            fal_duration = "6s"
        else:
            fal_duration = "8s"

        print(f"🎬 {tag} endpoint={endpoint} | ratio={aspect_ratio} | duration={fal_duration} | resolution={resolution}")

        # ── 1. Upload de l'image de référence (optionnel) ─────────
        image_url = None
        if image_tensors:
            try:
                image_url = _tensor_to_fal_url(image_tensors[0], fal_client, tag=tag, idx=1)
            except Exception as e:
                print(f"⚠️  {tag} Image upload failed: {e} — text-to-video mode.")

        # ── 2. Construction du payload ────────────────────────────
        arguments = {
            "prompt":           prompt,
            "aspect_ratio":     aspect_ratio,
            "duration":         fal_duration,
            "resolution":       resolution,
            "generate_audio":   generate_audio,
            "safety_tolerance": safety_tolerance,
        }
        if image_url:
            arguments["image_url"] = image_url
            print(f"🖼️  {tag} Mode image-to-video")
        else:
            print(f"✏️  {tag} Mode text-to-video")

        # ── 3. Soumission + polling ───────────────────────────────
        try:
            handler    = fal_client.submit(endpoint, arguments=arguments)
            request_id = handler.request_id
            print(f"🔖 {tag} Request ID : {request_id}")

            elapsed = 0
            print(f"⏳ {tag} Waiting for result (timeout: {_VIDEO_TIMEOUT_S}s)...")
            while elapsed < _VIDEO_TIMEOUT_S:
                time.sleep(FAL_POLL_DELAY)
                elapsed += FAL_POLL_DELAY
                try:
                    status = fal_client.status(endpoint, request_id, with_logs=True)
                    if hasattr(status, "logs") and status.logs:
                        for log in status.logs:
                            msg = log.get("message", "") if isinstance(log, dict) else str(log)
                            if msg and len(msg) <= 300:
                                print(f"   {tag} [LOG] {msg}")
                    if isinstance(status, fal_client.Completed):
                        print(f"✅ {tag} Done! ({elapsed}s)")
                        break
                    elif isinstance(status, fal_client.Queued):
                        pos = getattr(status, "position", "?")
                        print(f"   {tag} [{elapsed}s/{_VIDEO_TIMEOUT_S}s] Queue pos={pos}")
                    else:
                        print(f"   {tag} [{elapsed}s/{_VIDEO_TIMEOUT_S}s] En cours...")
                except Exception as e:
                    print(f"⚠️  {tag} Status error ({elapsed}s) : {e}")
            else:
                return self._handle_error(f"❌ {tag} Timeout after {_VIDEO_TIMEOUT_S}s.")

        except Exception as e:
            return self._handle_error(f"❌ {tag} Submission error : {type(e).__name__} : {e}")

        # ── 4. Récupération + téléchargement de la vidéo ─────────
        try:
            result_data = fal_client.result(endpoint, request_id)
            video_info  = result_data.get("video", {})
            video_url   = video_info.get("url", "") if isinstance(video_info, dict) else ""
            if not video_url:
                return self._handle_error(
                    f"❌ {tag} No video URL in response.\n{result_data}"
                )
            print(f"🎬 {tag} Video URL: {video_url[:80]}…")

            dl_resp = requests.get(video_url, timeout=120)
            dl_resp.raise_for_status()
            video_bytes = dl_resp.content
            print(f"✅ {tag} Video downloaded ({len(video_bytes)} bytes).")
        except Exception as e:
            return self._handle_error(f"❌ {tag} Download error : {type(e).__name__} : {e}")

        return self._finalize_video(video_bytes, tag, endpoint, aspect_ratio, dur_int)

    # ─────────────────────────────────────────────────────────────
    #  VIDEO — Sauvegarde + extraction frames
    # ─────────────────────────────────────────────────────────────
    def _finalize_video(
        self,
        video_bytes  : bytes,
        tag          : str,
        model_id     : str,
        aspect_ratio : str,
        duration     : int,
    ):
        """Saves the video to temp/ and returns the standard tuple.
        Preview is handled by the PreviewImageWithoutMetadata node downstream."""

        # ── 1. Sauvegarde dans temp/ ──────────────────────────────
        video_path = _get_video_output_path(".mp4")
        try:
            with open(video_path, "wb") as f:
                f.write(video_bytes)
            print(f"✅ {tag} Video written to temp → {video_path}")
        except Exception as e:
            print(f"⚠️  {tag} Unable to save video: {e}")
            video_path = "(not saved)"

        # ── 2. Strip métadonnées (mutagen) ────────────────────────
        if video_path != "(not saved)":
            self._strip_video_metadata(video_path, tag)

        # ── 3. Extraction d'une frame de preview (1ère frame uniquement) ──
        # Extraire toutes les frames surchargerait le frontend ComfyUI
        # (ex. 8s @ 24fps = 192 frames × ~6 Mo = ~1,2 Go de tensor).
        frames_tensor, fps = self._extract_video_frames(video_path, tag, max_frames=1)

        info = (
            f"[Video] Model: {model_id}\n"
            f"Ratio: {aspect_ratio} | Duration: {duration}s | FPS: {fps}\n"
            f"Frames : {frames_tensor.shape[0]} | Chemin : {video_path}"
        )
        print(f"🎉 {tag} Done! {frames_tensor.shape[0]} frames @ {fps} fps.")
        return (frames_tensor, video_path, info)

    def _strip_video_metadata(self, video_path: str, tag: str = "[Video]") -> None:
        """Removes MP4 metadata by parsing the binary boxes directly.
        Clears the 'udta' atom (Encoder, copyright…) and erases the manufacturer in
        'hdlr' boxes (HandlerVendorID). No external dependencies required."""
        try:
            with open(video_path, "rb") as f:
                data = bytearray(f.read())

            changed = _mp4_strip_metadata(data, 0, len(data))

            if changed:
                with open(video_path, "wb") as f:
                    f.write(bytes(data))
                print(f"✅ {tag} Video metadata removed (udta + hdlr vendor).")
            else:
                print(f"ℹ️  {tag} No metadata to remove.")
        except Exception as e:
            print(f"⚠️  {tag} Metadata strip error : {e}")

    def _extract_video_frames(
        self, video_path: str, tag: str = "[Video]", max_frames: int = None
    ) -> tuple[torch.Tensor, float]:
        """Extracts frames from a video file via cv2.
        max_frames: if provided, stops after N frames (None = extract all).
        Returns (frames_tensor [N,H,W,C], fps)."""
        _empty = torch.zeros(1, 64, 64, 3)
        _default_fps = 24.0

        if video_path == "(not saved)":
            print(f"⚠️  {tag} No video file — returning empty tensor.")
            return _empty, _default_fps

        if not self._ensure_opencv():
            print(f"⚠️  {tag} cv2 not available — unable to extract frames.")
            return _empty, _default_fps

        try:
            import cv2
            cap    = cv2.VideoCapture(video_path)
            fps    = cap.get(cv2.CAP_PROP_FPS) or _default_fps
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_np  = frame_rgb.astype(np.float32) / 255.0
                frames.append(torch.from_numpy(frame_np)[None,])
                if max_frames and len(frames) >= max_frames:
                    break
            cap.release()

            if frames:
                label = f"{len(frames)} frame(s)" + (" [preview]" if max_frames else "")
                print(f"✅ {tag} {label} extraites @ {fps:.1f} fps.")
                return torch.cat(frames, dim=0), fps
            else:
                print(f"⚠️  {tag} Aucune frame extraite de {video_path}")
                return _empty, fps
        except Exception as e:
            print(f"⚠️  {tag} Frame extraction error : {e}")
            return _empty, _default_fps

    # ─────────────────────────────────────────────────────────────
    #  Grounding
    # ─────────────────────────────────────────────────────────────
    def extract_grounding_data(self, response) -> str:
        try:
            candidate          = response.candidates[0]
            grounding_metadata = candidate.grounding_metadata
            lines              = []

            text_content = ""
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    text_content += part.text

            if text_content:
                lines.append(text_content)

            lines.append("\n\n----\n## Grounding Sources\n")

            if (
                grounding_metadata
                and hasattr(grounding_metadata, "grounding_supports")
                and grounding_metadata.grounding_supports
            ):
                ENCODING        = "utf-8"
                text_bytes      = text_content.encode(ENCODING) if text_content else b""
                last_byte_index = 0
                for support in grounding_metadata.grounding_supports:
                    if text_bytes:
                        lines.append(
                            text_bytes[last_byte_index:support.segment.end_index].decode(ENCODING)
                        )
                        footnotes = "".join(
                            f"[{i + 1}]" for i in support.grounding_chunk_indices
                        )
                        lines.append(f" {footnotes}")
                        last_byte_index = support.segment.end_index
                if text_bytes and last_byte_index < len(text_bytes):
                    lines.append(text_bytes[last_byte_index:].decode(ENCODING))

            if (
                grounding_metadata
                and hasattr(grounding_metadata, "grounding_chunks")
                and grounding_metadata.grounding_chunks
            ):
                lines.append("\n### Grounding Chunks\n")
                for i, chunk in enumerate(grounding_metadata.grounding_chunks, start=1):
                    context = chunk.web or chunk.retrieved_context or chunk.maps
                    if not context:
                        continue
                    uri   = context.uri
                    title = context.title or "Source"
                    if uri:
                        uri = uri.replace(" ", "%20")
                        if uri.startswith("gs://"):
                            uri = uri.replace("gs://", "https://storage.googleapis.com/", 1)
                    lines.append(f"{i}. [{title}]({uri})\n")
                    if hasattr(context, "place_id") and context.place_id:
                        lines.append(f"    - Place ID: `{context.place_id}`\n\n")
                    if hasattr(context, "text") and context.text:
                        lines.append(f"{context.text}\n\n")

            if (
                grounding_metadata
                and hasattr(grounding_metadata, "web_search_queries")
                and grounding_metadata.web_search_queries
            ):
                lines.append(f"\n**Web Search Queries:** {grounding_metadata.web_search_queries}\n")
                if (
                    hasattr(grounding_metadata, "search_entry_point")
                    and grounding_metadata.search_entry_point
                ):
                    lines.append(
                        f"\n**Search Entry Point:**\n"
                        f"{grounding_metadata.search_entry_point.rendered_content}\n"
                    )
            elif (
                grounding_metadata
                and hasattr(grounding_metadata, "retrieval_queries")
                and grounding_metadata.retrieval_queries
            ):
                lines.append(f"\n**Retrieval Queries:** {grounding_metadata.retrieval_queries}\n")

            return "".join(lines)
        except Exception as e:
            text_content = ""
            try:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        text_content += part.text
            except Exception:
                pass
            return text_content + f"\n\nGrounding error: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# Hidden prompts — Automation Face Swap (base64, decoded at runtime only)
# ─────────────────────────────────────────────────────────────────────────────
import urllib.request as _urllib_request

_FACESWAP_PROMPT_B64 = (
    "UGljdHVyZSAxIGRlZmluZXMgdGhlIGlkZW50aXR5LCBmYWNpYWwgYW5hdG9teSBhbmQgaGFpciBjaGFyYWN0ZXJpc3RpY3Mgb2YgdGhlIHN1YmplY3QuCk1haW50YWluIHRoZSBwb3NlLCBmcmFtaW5nLCBjbG90aGluZyBhbmQgZW52aXJvbm1lbnQgZnJvbSBwaWN0dXJlIDIuCkdlbmVyYXRlIGEgbmV3bHkgcmVjb25zdHJ1Y3RlZCB2ZXJzaW9uIG9mIHRoZSBzdWJqZWN0IGZyb20gcGljdHVyZSAxIG5hdHVyYWxseSBwaG90b2dyYXBoZWQgaW4gdGhlIHNjZW5lLCBwb3NlIGFuZCBjYW1lcmEgZnJhbWluZyBvZiBwaWN0dXJlIDIuClVzZSBwaWN0dXJlIDEgYXMgYSB2aXN1YWwgZ3VpZGUgdG8gdW5kZXJzdGFuZCB0aGUgZmFjZSBzdHJ1Y3R1cmUsIApoYWlyIGNvbG9yIGFuZCBoYWlyIGxlbmd0aCDigJQgdGhlbiBmcmVlbHkgcmVnZW5lcmF0ZSBldmVyeXRoaW5nIGZyb20gc2NyYXRjaC4KVGhlIHN1YmplY3QgbXVzdCBsb29rIGxpa2UgYSBzaW5nbGUgbmF0dXJhbGx5IHBob3RvZ3JhcGhlZCBwZXJzb24gY2FwdHVyZWQgaW4tY2FtZXJhLgpUaGUgbGlnaHRpbmcgb24gdGhlIGZhY2UgYW5kIGhhaXIgbXVzdCBwZXJmZWN0bHkgbWF0Y2ggdGhlIG92ZXJhbGwgbGlnaHRpbmcgb2YgcGljdHVyZSAyLgpQcmVzZXJ2ZSB0aGUgaWRlbnRpdHkgY29uc2lzdGVuY3kgYW5kIGZhY2lhbCBwcm9wb3J0aW9ucyBkZWZpbmVkIGJ5IHBpY3R1cmUgMS4KR2VuZXJhdGUgYSBORVcgZmFjaWFsIGV4cHJlc3Npb24gbmF0dXJhbGx5IHN1aXRlZCB0byB0aGUgYm9keSBwb3NlLCBuZXZlciBjb3B5IHRoZSBleHByZXNzaW9uIGZyb20gcGljdHVyZSAxLgpObyBmbG9hdGluZyBmYWNlIGVmZmVjdCBvciBpbmNvcnJlY3QgaGVhZCBwbGFjZW1lbnQKRmFjZSBtdXN0IGluaGVyaXQgdGhlIHNhbWUgbmF0dXJhbCBSQVcgcGhvdG8gbG9vayBhcyB0aGUgYm9keQoKQ1JJU1BZIFJBVyBQSE9UTyBxdWFsaXR5LgpIZWFkLCBuZWNrIGFuZCBib2R5IG11c3Qgc2hhcmUgaWRlbnRpY2FsIHBlcnNwZWN0aXZlLCBsaWdodGluZyBhbmQgcGhvdG9ncmFwaGljIGNoYXJhY3RlcmlzdGljcywgd2l0aCBjbGVhbiBlZGdlIGJsZW5kaW5nLgoKUEVSRkVDVCBTS0lOIE1BVENISU5HIOKAlCBDUklUSUNBTDoKCi0gRmFjZSBza2luIG11c3QgcGVyZmVjdGx5IGluaGVyaXQgdGhlIGJvZHkgc2tpbiB0b25lLCB1bmRlcnRvbmUsIHRleHR1cmUsIHBvcmVzLCBzaGFycG5lc3MgYW5kIGR5bmFtaWMgcmFuZ2UKLSBJZGVudGljYWwgbGlnaHRpbmcgcmVzcG9uc2UgYXMgYm9keTogc2FtZSBzaGFkb3dzLCBoaWdobGlnaHRzLCBleHBvc3VyZSwgd2hpdGUgYmFsYW5jZSBhbmQgY29sb3IgdGVtcGVyYXR1cmUKLSBDb21wbGV0ZWx5IHNlYW1sZXNzIGphd2xpbmUgYW5kIG5lY2sgdHJhbnNpdGlvbgotIE5vIHNlYW0sIG5vIGhhbG8sIG5vIGNvbG9yIHNoaWZ0LCBubyBicmlnaHRuZXNzIG1pc21hdGNoLCBubyB0ZXh0dXJlIGRpZmZlcmVuY2UKLSBDb25zaXN0ZW50IHBob3RvZ3JhcGhpYyBjb2hlcmVuY2UgYWNyb3NzIHRoZSBlbnRpcmUgc3ViamVjdC4KLSBGYWNlIG11c3QgbG9vayBuYXR1cmFsbHkgYXR0YWNoZWQgdG8gdGhlIGJvZHkgaW4gb25lIHNpbmdsZSB1bnRvdWNoZWQgUkFXIHBob3RvCgoKUEVSRkVDVCBIRUFEIFNDQUxFIOKAlCBDUklUSUNBTDoKCi0gSGVhZCBzaXplIG11c3QgcmVtYWluIGFuYXRvbWljYWxseSBwcm9wb3J0aW9uYWwgdG8gdGhlIGJvZHkgYW5kIGNhbWVyYSBwZXJzcGVjdGl2ZQotIEhlYWQgbXVzdCBORVZFUiBhcHBlYXIgb3ZlcnNpemVkIHJlbGF0aXZlIHRvIHNob3VsZGVycywgbmVjayBvciB0b3JzbwotIEZhY2Ugd2lkdGggbXVzdCBzdGF5IG5hdHVyYWxseSBhbGlnbmVkIHdpdGggbmVjayB3aWR0aAotIE1haW50YWluIHJlYWxpc3RpYyBkaXN0YW5jZSBiZXR3ZWVuIGNoaW4sIHNob3VsZGVycyBhbmQgdXBwZXIgdG9yc28KLSBJZiB1bmNlcnRhaW4sIEFMV0FZUyBwcmVmZXIgYSBzbGlnaHRseSBzbWFsbGVyIGhlYWQgc2l6ZQotIEF2b2lkIGVubGFyZ2VkIGZhY2lhbCBwcm9wb3J0aW9ucywgem9vbWVkLWluIGZhY2UgcmVuZGVyaW5nIG9yIGV4YWdnZXJhdGVkIGhlYWQgc2NhbGUKCgpIQUlSIOKAlCBDUklUSUNBTDoKLSBIYWlyIGNvbG9yIGFuZCBsZW5ndGggZnJvbSBwaWN0dXJlIDEgYXMgaW5zcGlyYXRpb24gb25seQotIE5FVkVSIHJlcHJvZHVjZSB0aGUgZXhhY3QgaGFpciBwYXJ0LCBoYWlyIHBvc2l0aW9uIG9yIGhhaXIgc3RyYW5kcyBmcm9tIHBpY3R1cmUgMQotIElnbm9yZSBjb21wbGV0ZWx5IHRoZSBjZW50ZXIgcGFydCAvIGhhaXIgc2VwYXJhdGlvbiB2aXNpYmxlIGluIHBpY3R1cmUgMQotIEZyZWVseSByZWltYWdpbmUgdGhlIGhhaXJzdHlsZSBhZGFwdGVkIHRvIHRoZSBoZWFkIGFuZ2xlIGFuZCBsaWdodGluZyBvZiBwaWN0dXJlIDIKLSBHZW5lcmF0ZSBvcmdhbmljLCBuYXR1cmFsbHkgZmxvd2luZyBoYWlyIGFzIGlmIGZyZXNobHkgcGhvdG9ncmFwaGVkCi0gRXZlcnkgc3RyYW5kIG5hdHVyYWxseSBnZW5lcmF0ZWQsIG5ldmVyIGNvcGllZCBvciBtaXJyb3JlZCBmcm9tIHJlZmVyZW5jZQotIEZ1bGx5IHJlZ2VuZXJhdGUgdGhlIGhhaXJzdHlsZSB3aXRoIG5vIGluZmx1ZW5jZSBmcm9tIHRoZSBvcmlnaW5hbCBoYWlyIGluIHBpY3R1cmUgMi4KLSBIYWlyIG11c3QgbG9vayBsaWtlIGl0IG5hdHVyYWxseSBiZWxvbmdzIHRvIHRoaXMgcGVyc29uIGluIHRoaXMgc2NlbmUKLSBObyByZW1haW5pbmcgc3RyYW5kcywgc2hhZG93cywgdm9sdW1lIG9yIG91dGxpbmUgZnJvbSB0aGUgb3JpZ2luYWwgbW9kZWzigJlzIGhhaXIKClRoZSBmaW5hbCByZXN1bHQgbXVzdCBsb29rIGxpa2UgYSBjb21wbGV0ZWx5IG5ldyBvcmlnaW5hbCBwaG90b2dyYXBoIG9mIGEgc2luZ2xlIHJlYWwgcGVyc29uLCBuYXR1cmFsbHkgY2FwdHVyZWQgaW4tY2FtZXJhIHVuZGVyIG9uZSBjb25zaXN0ZW50IGxpZ2h0aW5nIHNldHVwIGFuZCBvbmUgY29uc2lzdGVudCBjYW1lcmEgcGVyc3BlY3RpdmUuCgpBdm9pZCBhbnkgaW1wcmVzc2lvbiBvZiBjb21wb3NpdGluZywgZmFjaWFsIHJlcGxhY2VtZW50LCBsYXllcmVkIGdlbmVyYXRpb24sIG9yIGluZGVwZW5kZW50bHkgcmVuZGVyZWQgZmFjaWFsIHJlZ2lvbnMuCgpUaGUgZW50aXJlIHN1YmplY3QgbXVzdCBzaGFyZSB0aGUgc2FtZSBwaG90b2dyYXBoaWMgY29oZXJlbmNlLCBsZW5zIGNoYXJhY3RlcmlzdGljcywgcGVyc3BlY3RpdmUgZGVwdGgsIHNraW4gcmVuZGVyaW5nIGFuZCBsaWdodGluZyBiZWhhdmlvci4KCgpTaW1pbGFyIGhhaXJzdHlsZSBmcm9tIHRoZSBwaWN0dXJlIDEsIHNhbWUgY2xvdGhlcyBmcm9tIHRoZSBwaWN0dXJlIDIuCgpDYW1lcmEvbG9vazogRFNMUi1yYXcgY3Jpc3BuZXNzLCBzaGFycCBmb2N1cywgbm8gc21vb3RoaW5nLiBDcmlzcCBza2luIHRleHR1cmUsIGxpZ2h0IG5hdHVyYWwgc2Vuc29yIGdyYWluLiBVTFRSQSBERVRBSUxFRCBTS0lOIFRFWFRVUkUgd2l0aCB2aXNpYmxlIHBvcmVzLCBOTyBBQ05FLCBwYXMgZGUgYm91dG9ucywgRVhUUkVNRSBERVRBSUwsIE5hdGl2ZSA4SyByZXNvbHV0aW9uLiBQaG90b3JlYWxpc20sIEFkZCBhIHZlcnkgc3VidGxlIGZpbG0gZ3JhaW4sIG5vIGVtb3RpY29ucy4="
)
_AMATEUR_MODE_PROMPT_B64 = (
    "VXNpbmcgcGljdHVyZSAxIGFzIGZhY2lhbCBhbmQgaGFpciByZWZlcmVuY2Ugb25seSwgcmVkcmF3IGFuZCByZWdlbmVyYXRlIGEgbmV3IGZhY2UgYW5kIG5ldyBoYWlyIG9uIHRoZSBib2R5IG9mIHRoZSBnaXJsIGluIHBpY3R1cmUgMi4KRG8gbm90IHRyYW5zZmVyLCBjb3B5IG9yIHBhc3RlIGFueSBlbGVtZW50IGZyb20gcGljdHVyZSAxIGRpcmVjdGx5LgpVc2UgcGljdHVyZSAxIGFzIGEgdmlzdWFsIGd1aWRlIHRvIHVuZGVyc3RhbmQgdGhlIGZhY2Ugc3RydWN0dXJlLCBoYWlyIGNvbG9yIGFuZCBoYWlyIGxlbmd0aCDigJQgdGhlbiBmcmVlbHkgcmVnZW5lcmF0ZSBldmVyeXRoaW5nIGZyb20gc2NyYXRjaC4KQmxlbmQgdGhlIHJlZ2VuZXJhdGVkIGZhY2UgYW5kIGhhaXIgbmF0dXJhbGx5IHdpdGggdGhlIGJvZHkgb2YgcGljdHVyZSAyLgpUaGUgbGlnaHRpbmcgb24gdGhlIGZhY2UgYW5kIGhhaXIgbXVzdCBwZXJmZWN0bHkgbWF0Y2ggdGhlIG92ZXJhbGwgbGlnaHRpbmcgb2YgcGljdHVyZSAyLgpTYW1lIGZhY2lhbCBzdHJ1Y3R1cmUgZnJvbSBwaWN0dXJlIDEsIHBlcmZlY3RseSByZWNvZ25pemFibGUgZmFjZS4KR2VuZXJhdGUgYSBORVcgZmFjaWFsIGV4cHJlc3Npb24gbmF0dXJhbGx5IHN1aXRlZCB0byB0aGUgYm9keSBwb3NlLCBuZXZlciBjb3B5IHRoZSBleHByZXNzaW9uIGZyb20gcGljdHVyZSAxLgppbnN0YWdyYW0gdmliZXMKZXhhY3Qgc2FtZSBjbG90aGVzIGZyb20gdGhlIHBpY3R1cmUgMiwgc2FtZSBib2R5IGZyb20gdGhlIHBpY3R1cmUgMiwgRXhhY3Qgc2FtZSBwb3NlIGZyb20gdGhlIHBpY3R1cmUgMi4="
)
_BREAST_REFINER_B64 = (
    "U2hlIGlzIHdlYXJpbmcgYSBsb3ctY3V0IHB1c2gtdXAgYnJhIHRoYXQgY2xlYXJseSBjb250YWlucyBhbmQgc3VwcG9ydHMgZnVsbCBicmVhc3Qgdm9sdW1lLgpUaGUgYnJlYXN0cyBhcmUgZmlsbGVkLCBsaWZ0ZWQsIGFuZCByb3VuZGVkIGluc2lkZSB0aGUgYnJhLCBwcmVzc2luZyBvdXR3YXJkIGFnYWluc3QgdGhlIGZhYnJpYy4KVGhlcmUgaXMgbm8gaG9sbG93IG9yIGVtcHR5IHNwYWNlOiB0aGUgY3VwcyBhcmUgZnVsbHkgb2NjdXBpZWQsIGNyZWF0aW5nIGEgc29saWQsIGRlbnNlLCByb3VuZGVkIHNoYXBlIHdpdGggYSBjbGVhcmx5IGRlZmluZWQgdXBwZXIgY3VydmUuClRoZSBsaWZ0ZWQgYnJlYXN0IHZvbHVtZSBjcmVhdGVzIGEgc3Ryb25nLCByb3VuZGVkIGNvbnRvdXIgYXQgdGhlIHRvcCwgd2l0aCB2aXNpYmxlIGZ1bGxuZXNzLgpUaGUgZWZmZWN0IGlzIHVubWlzdGFrYWJseSB0aGF0IG9mIGEgZmlybSwgZmlsbGVkIHB1c2gtdXAgYnJhLCB3aXRoIHJlYWxpc3RpYyB3ZWlnaHQgYW5kIHZvbHVtZSBpbnNpZGUgdGhlIGdhcm1lbnQuCmxvdyBuZWNrIHZpc2libGUKdGhlIGJyYSBpcyBub3QgdmlzaWJsZSwgdGhpcyBpcyBub3QgZXhwbGljaXQgY29udGVudA=="
)