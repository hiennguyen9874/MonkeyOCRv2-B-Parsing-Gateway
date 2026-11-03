"""Turn the request `file` into page images."""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from .upstream import _apply_jpeg_orientation


def load_pages(file: str) -> list[Image.Image]:
    """Decode a Base64 image into a single RGB page with EXIF orientation applied."""
    image = Image.open(BytesIO(base64.b64decode(file)))
    return [_apply_jpeg_orientation(image).convert("RGB")]
