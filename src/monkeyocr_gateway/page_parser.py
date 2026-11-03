"""Parse one page image into Khối."""

from __future__ import annotations

import asyncio

from PIL import Image

from .upstream import (
    ALL_PROMPT,
    LAYOUT_MAX_TOKENS,
    LAYOUT_MIN_PIXELS,
    RECOGNITION_MAX_TOKENS,
    _build_page_tasks,
    parse_end2end_output,
    parse_layout_output,
)
from .vllm_client import VLLMClient


async def gather_or_cancel(coroutines) -> list:
    """Results in input order whatever order they finish in. On the first error,
    cancel and drain the others, then re-raise that error unchanged."""
    tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def parse_page(image: Image.Image, mode: str, vllm: VLLMClient) -> list[dict]:
    """mode is one of config.PARSING_MODES.

    Returns Khối in reading order as {"bbox": [x1, y1, x2, y2] page pixels, "label", "content"}.
    Picture Khối also carry "crop": the bbox region of the full-resolution page image.
    """
    if mode == "end-to-end":
        return await _parse_end_to_end(image, vllm)
    return await _parse_two_stage(image, vllm)


async def _parse_end_to_end(image: Image.Image, vllm: VLLMClient) -> list[dict]:
    """Chế độ end-to-end: one inference returns label, bbox and content of every Khối."""
    raw = await vllm.chat(image, ALL_PROMPT["END2END"])  # Upstream sets no max_tokens here
    records, _layouts = parse_end2end_output(raw, image.size)
    return [
        _khoi(record["bbox"], record["label"], record["content"], image)
        for record in records
    ]


async def _parse_two_stage(image: Image.Image, vllm: VLLMClient) -> list[dict]:
    """Chế độ hai giai đoạn, as Upstream's ServicePipelinePool._parse_page does it:
    layout on the whole page, then recognize each prompted Khối cropped from the
    full-resolution page. Khối whose label has no prompt keep empty content."""
    raw_layout = await vllm.chat(
        image,
        ALL_PROMPT["LAYOUT"],
        min_pixels=LAYOUT_MIN_PIXELS,
        max_tokens=LAYOUT_MAX_TOKENS,
    )
    tasks = _build_page_tasks(0, image, parse_layout_output(raw_layout, image.size))

    async def recognize(task: dict) -> str:
        if not task["need_infer"]:
            return ""
        return await vllm.chat(task["image"], task["question"], max_tokens=RECOGNITION_MAX_TOKENS)

    raws = await gather_or_cancel([recognize(task) for task in tasks])
    return [
        _khoi(task["bbox"], task["label"], (raw or "").strip(), image)
        for task, raw in zip(tasks, raws)
    ]


def _khoi(bbox: list[int], label: str, content: str, page: Image.Image) -> dict:
    block = {"bbox": bbox, "label": label, "content": content}
    if label == "Picture":
        # bbox is already clamped to integer page pixels by the Upstream helpers.
        block["crop"] = page.crop(tuple(bbox))
    return block
