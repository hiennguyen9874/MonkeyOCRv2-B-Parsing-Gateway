"""Pure helpers ported from Upstream MonkeyOCRv2.

Source: references/MonkeyOCRv2/parsing/core_runner.py at revision 106914e.
Upstream imports torch and its preprocessor at module level, so the helpers are
copied here instead of imported (ADR-0003). Keep them behaviourally identical;
tests/test_upstream_parity.py reuses Upstream's test cases to catch drift.
"""

from __future__ import annotations

import ast
import base64
from io import BytesIO

from PIL import Image, ImageOps

UPSTREAM_REVISION = "106914e"

# core_runner.ALL_PROMPT (only the end-to-end prompt is used so far).
ALL_PROMPT = {
    "END2END": "List the document elements in reading order, including their categories, coordinates, and the content of each element.",
}


def image_to_png_data_uri(image: Image.Image) -> str:
    """core_runner.image_to_png_data_uri"""
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _extract_balanced_blocks(text: str, left: str, right: str):
    """core_runner._extract_balanced_blocks"""
    blocks = []
    depth = 0
    start = -1
    for index, char in enumerate(text):
        if char == left:
            if depth == 0:
                start = index
            depth += 1
        elif char == right and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                blocks.append(text[start:index + 1])
                start = -1
    return blocks


def _deduplicate_strings(values):
    """core_runner._deduplicate_strings"""
    return list(dict.fromkeys(values))


def _extract_tolerant_list_blocks(text: str):
    """core_runner._extract_tolerant_list_blocks"""
    blocks = _extract_balanced_blocks(text, "[", "]")
    first = text.find("[")
    if first != -1:
        tail = text[first:].strip()
        missing = tail.count("[") - tail.count("]")
        if tail and missing > 0:
            blocks.append(tail + ("]" * missing))
    return _deduplicate_strings(blocks)


def _extract_tolerant_dict_blocks(text: str):
    """core_runner._extract_tolerant_dict_blocks"""
    blocks = _extract_balanced_blocks(text, "{", "}")
    for index, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        end = None
        for cursor in range(index, len(text)):
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
                if depth == 0:
                    end = cursor + 1
                    break
        blocks.append(
            text[index:end]
            if end is not None
            else text[index:] + ("}" * max(depth, 1))
        )
    return _deduplicate_strings(blocks)


def _parse_tolerant_items(text: str, normalize_item):
    """core_runner._parse_tolerant_items"""
    text = (text or "").strip()
    if not text:
        return []

    def normalize_list(value):
        if not isinstance(value, list):
            return []
        return [item for raw in value if (item := normalize_item(raw)) is not None]

    try:
        complete = normalize_list(ast.literal_eval(text))
        if complete:
            return complete
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        pass

    best = []
    for block in _extract_tolerant_list_blocks(text):
        try:
            current = normalize_list(ast.literal_eval(block))
            if len(current) > len(best):
                best = current
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            continue

    dict_items = []
    for block in _extract_tolerant_dict_blocks(text):
        try:
            item = normalize_item(ast.literal_eval(block))
            if item is not None:
                dict_items.append(item)
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            continue
    return dict_items if len(dict_items) > len(best) else best


def _map_bbox_to_image(bbox, width: int, height: int):
    """core_runner._map_bbox_to_image: 0-1000 model space -> integer pixels."""
    x1, y1, x2, y2 = bbox
    x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
    y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1 = max(0, min(int(round(x1)), width - 1 if width > 0 else 0))
    y1 = max(0, min(int(round(y1)), height - 1 if height > 0 else 0))
    x2 = max(x1 + 1, min(int(round(x2)), width))
    y2 = max(y1 + 1, min(int(round(y2)), height))
    return [x1, y1, x2, y2]


def _normalize_model_item(item, include_content: bool):
    """core_runner._normalize_model_item"""
    if not isinstance(item, dict) or "bbox" not in item or "label" not in item:
        return None
    bbox = item["bbox"]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        bbox = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    normalized = {
        "bbox": bbox,
        "label": item["label"] if isinstance(item["label"], str) else str(item["label"]),
    }
    if include_content:
        content = item.get("content", "")
        normalized["content"] = content if isinstance(content, str) else str(content or "")
    return normalized


def parse_end2end_output(text: str, image_size: tuple[int, int]) -> tuple[list[dict], list[dict]]:
    """core_runner.parse_end2end_output"""
    width, height = image_size
    records = []
    layouts = []
    items = _parse_tolerant_items(
        text,
        lambda item: _normalize_model_item(item, include_content=True),
    )
    for block_idx, item in enumerate(items):
        bbox = _map_bbox_to_image(item["bbox"], width, height)
        label = item["label"]
        records.append({
            "bbox": bbox,
            "label": label,
            "content": (item.get("content") or "").strip(),
            "_block_idx": block_idx,
        })
        layouts.append({"bbox": bbox, "label": label})
    return records, layouts


def _apply_jpeg_orientation(img: Image.Image) -> Image.Image:
    """core_runner._apply_jpeg_orientation, without the path-based JPEG check
    (the gateway never loads from a path or URL string)."""
    if (getattr(img, "format", None) or "").upper() == "JPEG":
        return ImageOps.exif_transpose(img)
    return img


def load_image(image: Image.Image, max_pixels: int | None = None, min_pixels: int | None = None) -> Image.Image:
    """core_runner.load_image, restricted to the PIL-image branch and the
    max_pixels/min_pixels arguments that MonkeyOCRv2_ServerParsing.batch_inference uses.
    """
    try:
        img = _apply_jpeg_orientation(image)
        img = img.convert("RGB")

        # resize image if too small
        if min_pixels and img.size[0] * img.size[1] < min_pixels:
            scale = (min_pixels / (img.size[0] * img.size[1])) ** 0.5
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.LANCZOS)

        # resize image if too large
        if max_pixels and img.size[0] * img.size[1] > max_pixels:
            scale = (max_pixels / (img.size[0] * img.size[1])) ** 0.5
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.LANCZOS)

        if max(img.size[0], img.size[1]) / min(img.size[0], img.size[1]) > 200:
            img = Image.new("RGB", (32, 32))
    except Exception:
        # Upstream silently substitutes a dummy image for anything it cannot load.
        img = Image.new("RGB", (32, 32))
    return img
