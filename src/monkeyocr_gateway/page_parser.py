"""Parse one page image into Khối."""

from __future__ import annotations

from PIL import Image

from .upstream import ALL_PROMPT, parse_end2end_output
from .vllm_client import VLLMClient


async def parse_page(image: Image.Image, vllm: VLLMClient) -> list[dict]:
    """Chế độ end-to-end: one inference returns label, bbox and content of every Khối.

    Returns Khối in reading order as {"bbox": [x1, y1, x2, y2] page pixels, "label", "content"}.
    """
    raw = await vllm.chat(image, ALL_PROMPT["END2END"])  # Upstream sets no max_tokens here
    records, _layouts = parse_end2end_output(raw, image.size)
    return [
        {"bbox": record["bbox"], "label": record["label"], "content": record["content"]}
        for record in records
    ]
