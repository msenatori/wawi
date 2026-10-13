"""Screen capture and base64-JPEG encoding for Wawi.

Uses mss.MSS() (not the deprecated mss.mss()).
thumbnail() only downsizes — the longest edge is guaranteed <= max_edge.
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import datetime
from pathlib import Path

import mss
from PIL import Image

log = logging.getLogger(__name__)

# Subdirectory screenshots land in when saved into a project worktree. A
# dedicated folder, because these files are throwaway context for an agent,
# not part of the project — they should be obvious to spot and easy to delete.
SHOTS_DIRNAME = ".wawi-shots"


def capture(monitor: int = 1) -> Image.Image:
    """Capture *monitor* and return a PIL Image (RGB).

    Args:
        monitor: Index into mss monitors list. 0 = union of all monitors;
                 1 = primary monitor (default). Values > 1 select additional
                 connected displays.

    Returns:
        PIL Image in RGB mode.
    """
    with mss.MSS() as sct:
        shot = sct.grab(sct.monitors[monitor])
        # mss returns BGRA; convert to RGB via PIL's raw decoder
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    return img


def save_jpeg(
    img: Image.Image,
    directory: Path | str,
    max_edge: int = 1568,
    jpeg_quality: int = 70,
    prefix: str = "pantalla",
) -> Path:
    """Write *img* into *directory* and return the path.

    Written to disk rather than passed inline because this is the screenshot
    the user asked FOR, not the one the model looks AT: a file survives the
    turn, can be opened later and dropped into anything else. The inline
    variant lives in :mod:`wawi.brain` and exists only for one message.

    The filename carries a timestamp so consecutive captures never overwrite
    each other — "the screenshot" is almost always the latest of several.

    Args:
        directory: Where to write. Created if missing.
        max_edge: Longest edge after downscaling; never upscales.
        jpeg_quality: 1-95.
        prefix: Filename prefix, before the timestamp.

    Returns:
        Path of the written file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    shrunk = img.copy()
    shrunk.thumbnail((max_edge, max_edge))
    if shrunk.mode != "RGB":
        shrunk = shrunk.convert("RGB")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = directory / f"{prefix}-{stamp}.jpg"
    shrunk.save(path, format="JPEG", quality=jpeg_quality)
    log.info("Screenshot saved: %s (%dx%d)", path, shrunk.width, shrunk.height)
    return path


# Defaults for the image the MODEL looks at. 1280 on the long edge is about
# 1,370 tokens on a 16:10 screen (width x height / 750), against ~2,050 at
# 1568, and the server rescales anything over ~1.15 MP anyway — so the extra
# pixels were uploaded and paid for in latency without adding detail. q65 keeps
# small text legible at that size. The file the user asks FOR (save_jpeg) is a
# different image and keeps its own, larger defaults.
INLINE_MAX_EDGE = 1280
INLINE_JPEG_QUALITY = 65


def encode_jpeg(
    img: Image.Image,
    max_edge: int = INLINE_MAX_EDGE,
    jpeg_quality: int = INLINE_JPEG_QUALITY,
) -> tuple[str, tuple[int, int]]:
    """Like :func:`to_base64_jpeg`, but also return the size actually encoded.

    The size is what the one-line capture note quotes, so it must be the
    post-resize one — the one the model saw — not the screen's.

    Returns:
        ``(base64_jpeg, (width, height))``.
    """
    # Work on a copy — thumbnail() mutates in place
    img = img.copy()
    img.thumbnail((max_edge, max_edge), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    return base64.b64encode(buf.getvalue()).decode("ascii"), img.size


def to_base64_jpeg(
    img: Image.Image,
    max_edge: int = INLINE_MAX_EDGE,
    jpeg_quality: int = INLINE_JPEG_QUALITY,
) -> str:
    """Resize *img* so the longest edge is at most *max_edge*, encode as JPEG, return base64.

    thumbnail() preserves aspect ratio and NEVER upscales, so the longest edge
    is guaranteed to be <= max_edge after the call.

    Args:
        img: Source PIL Image.
        max_edge: Maximum pixels on the longest edge after resize.
        jpeg_quality: JPEG quality parameter (1-95).

    Returns:
        Base64-encoded JPEG as a plain ASCII string (no line breaks).
    """
    return encode_jpeg(img, max_edge, jpeg_quality)[0]
