"""Parse one page image into Khối."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from PIL import Image

from .upstream import (
    ALL_PROMPT,
    LAYOUT_MAX_TOKENS,
    LAYOUT_MIN_PIXELS,
    RECOGNITION_MAX_TOKENS,
    _build_page_tasks,
    _should_retry_repeat_output,
    parse_end2end_output,
    parse_layout_output,
    repeat_retry_sampling,
)
from .vllm_client import GenerationOptions, VLLMClient


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


async def parse_page(
    image: Image.Image,
    mode: str,
    vllm: VLLMClient,
    options: GenerationOptions,
    repeat_retries: int = 0,
) -> list[dict]:
    """mode is one of config.PARSING_MODES. repeat_retries > 0 turns on Upstream's
    --retry-repeat for recognition and end-to-end output (layout is never retried).

    Returns Khối in reading order as {"bbox": [x1, y1, x2, y2] page pixels, "label", "content"}.
    """
    if mode == "end-to-end":
        return await _parse_end_to_end(image, vllm, options, repeat_retries)
    return await _parse_two_stage(image, vllm, options, repeat_retries)


async def _chat_with_repeat_retry(
    vllm: VLLMClient,
    image: Image.Image,
    prompt: str,
    options: GenerationOptions,
    repeat_retries: int,
    max_tokens: int | None = None,
) -> str:
    """Upstream _recognize_one_block: while the output ends in repeated tokens, ask
    again with rising temperature and top_p 0.95, at most repeat_retries times."""
    raw = await vllm.chat(image, prompt, options, max_tokens=max_tokens)
    for retry in range(repeat_retries):
        if not _should_retry_repeat_output(raw):
            break
        temperature, top_p = repeat_retry_sampling(retry)
        retry_options = replace(options, temperature=temperature, top_p=top_p)
        raw = await vllm.chat(image, prompt, retry_options, max_tokens=max_tokens)
    return raw


async def _parse_end_to_end(
    image: Image.Image, vllm: VLLMClient, options: GenerationOptions, repeat_retries: int
) -> list[dict]:
    """Chế độ end-to-end: one inference returns label, bbox and content of every Khối."""
    # Upstream sets no max_tokens here.
    raw = await _chat_with_repeat_retry(vllm, image, ALL_PROMPT["END2END"], options, repeat_retries)
    records, _layouts = parse_end2end_output(raw, image.size)
    return [
        {"bbox": record["bbox"], "label": record["label"], "content": record["content"]}
        for record in records
    ]


async def _parse_two_stage(
    image: Image.Image, vllm: VLLMClient, options: GenerationOptions, repeat_retries: int
) -> list[dict]:
    """Chế độ hai giai đoạn, as Upstream's ServicePipelinePool._parse_page does it:
    layout on the whole page, then recognize each prompted Khối cropped from the
    full-resolution page. Khối whose label has no prompt keep empty content."""
    raw_layout = await vllm.chat(
        image,
        ALL_PROMPT["LAYOUT"],
        options,
        min_pixels=LAYOUT_MIN_PIXELS,
        max_tokens=LAYOUT_MAX_TOKENS,
    )
    tasks = _build_page_tasks(0, image, parse_layout_output(raw_layout, image.size))

    async def recognize(task: dict) -> str:
        if not task["need_infer"]:
            return ""
        return await _chat_with_repeat_retry(
            vllm, task["image"], task["question"], options, repeat_retries, RECOGNITION_MAX_TOKENS
        )

    raws = await gather_or_cancel([recognize(task) for task in tasks])
    return [
        {"bbox": task["bbox"], "label": task["label"], "content": (raw or "").strip()}
        for task, raw in zip(tasks, raws)
    ]
