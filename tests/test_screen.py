"""Smoke tests for wawi/screen.py (Acceptance Scenario F1-S8).

Run with: python -m pytest tests/test_screen.py -v

test_capture_returns_image is skipped if Screen Recording permission is denied
(mss raises an exception or returns a black image on macOS without permission).
"""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image


def test_capture_returns_image() -> None:
    """screen.capture() returns a PIL Image without raising.

    Marked xfail if mss raises (Screen Recording may not be granted).
    """
    from wawi import screen

    try:
        img = screen.capture(monitor=1)
    except Exception as exc:
        pytest.xfail(f"Screen Recording permission likely denied: {exc}")
        return

    assert isinstance(img, Image.Image), "capture() must return a PIL Image"
    assert img.width > 0 and img.height > 0, "Image must have positive dimensions"


def test_to_base64_jpeg_respects_max_edge() -> None:
    """to_base64_jpeg() must ensure the longest edge is <= max_edge after resize."""
    from wawi import screen

    # Synthetic image larger than max_edge on both axes
    src = Image.new("RGB", (3000, 2000), color=(128, 64, 32))
    max_edge = 1568

    b64 = screen.to_base64_jpeg(src, max_edge=max_edge, jpeg_quality=70)

    # Decode and verify dimensions
    data = base64.b64decode(b64)
    result = Image.open(io.BytesIO(data))
    longest = max(result.width, result.height)
    assert longest <= max_edge, (
        f"Longest edge {longest} exceeds max_edge {max_edge}"
    )


def test_to_base64_jpeg_returns_nonempty_string() -> None:
    """to_base64_jpeg() must return a non-empty base64 string."""
    from wawi import screen

    src = Image.new("RGB", (100, 100), color=(0, 128, 255))
    b64 = screen.to_base64_jpeg(src, max_edge=1568, jpeg_quality=70)

    assert isinstance(b64, str), "Result must be a str"
    assert len(b64) > 0, "Result must be non-empty"
    # Verify it is valid base64
    decoded = base64.b64decode(b64)
    assert len(decoded) > 0, "Decoded bytes must be non-empty"


def test_to_base64_jpeg_does_not_upscale() -> None:
    """thumbnail() must not upscale images smaller than max_edge."""
    from wawi import screen

    small = Image.new("RGB", (100, 75), color=(255, 0, 0))
    b64 = screen.to_base64_jpeg(small, max_edge=1568, jpeg_quality=70)

    data = base64.b64decode(b64)
    result = Image.open(io.BytesIO(data))
    # JPEG decode may round by 1 pixel in extreme edge cases; use <=
    assert result.width <= 100 and result.height <= 75, (
        f"Image must not be upscaled: got {result.width}x{result.height}, "
        f"expected <= 100x75"
    )


def test_defaults_shrink_a_retina_screen_to_1280() -> None:
    """The image the model looks at defaults to 1280 on the long edge.

    2880x1800 is a 15" Retina panel. At 1568 it was ~2,050 tokens per look;
    at 1280 it is 1280x800, ~1,370 (width x height / 750).
    """
    from wawi import screen

    src = Image.new("RGB", (2880, 1800), color=(10, 20, 30))

    result = Image.open(io.BytesIO(base64.b64decode(screen.to_base64_jpeg(src))))

    assert max(result.size) <= 1280
    assert result.size == (1280, 800)


def test_encode_jpeg_reports_the_size_it_actually_encoded() -> None:
    """The capture note quotes this size, so it must be the post-resize one."""
    from wawi import screen

    src = Image.new("RGB", (2880, 1800), color=(10, 20, 30))

    b64, size = screen.encode_jpeg(src)

    assert size == Image.open(io.BytesIO(base64.b64decode(b64))).size == (1280, 800)
