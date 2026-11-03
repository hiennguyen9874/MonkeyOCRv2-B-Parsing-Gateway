"""Build PaddleOCR-VL Kết quả trang from Khối.

Both parsing modes go through this module, so the same Khối gives the same output.
"""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from .upstream import otsl_to_html, process_formula

# Nhãn Monkey -> Nhãn Paddle. Labels not listed keep their Nhãn Monkey.
PADDLE_LABELS = {
    "Text": "text",
    "List-item": "text",
    "Title": "doc_title",
    "Section-header": "paragraph_title",
    "Page-header": "header",
    "Page-footer": "footer",
    "Caption": "figure_title",
    "Table": "table",
    "Formula": "formula",
    "Picture": "image",
    "Footnote": "footnote",
}

# PaddleOCR-VL's default, used when the request sends markdownIgnoreLabels: null.
DEFAULT_MARKDOWN_IGNORE_LABELS = [
    "number",
    "footnote",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "aside_text",
]


def _collapse_soft_newlines(text: str) -> str:
    # PaddleX markdown_format_funcs._collapse_soft_newlines: a heading is one line.
    return text.replace("-\n", "").replace("\n", " ")


def _format_block(label: str, raw: str) -> tuple[str, str]:
    """Nhãn Paddle + raw Monkey content -> (block_content, Markdown fragment)."""
    if label == "table":
        html = otsl_to_html(raw)
        return html, html
    if label == "formula":
        # Markdown follows Upstream _format_block_fields: the equation number
        # extracted by process_formula goes after the closing $$.
        latex, number = process_formula(raw)
        markdown = "$$\n" + latex + "\n$$"
        if number:
            markdown += "\n" + number
        return latex, markdown
    if label == "image":
        # No OCR of image content; build_page_result references the crop instead.
        return "", ""
    if not raw:
        return "", ""
    if label == "doc_title":
        return raw, "# " + _collapse_soft_newlines(raw)
    if label == "paragraph_title":
        return raw, "## " + _collapse_soft_newlines(raw)
    return raw, raw


def _image_markdown(path: str, bbox: list[int], page_width: int) -> str:
    # PaddleOCR-VL serving default (prettifyMarkdown=true): PaddleX
    # format_image_scaled_by_html wrapped by format_centered_by_html.
    scale = int((bbox[2] - bbox[0]) / page_width * 100)
    return f'<div style="text-align: center;"><img src="{path}" alt="Image" width="{scale}%" /></div>'


def _jpeg_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_page_result(
    page: Image.Image,
    blocks: list[dict],
    model_settings: dict,
    page_count: int | None,
    page_index: int,
    return_markdown_images: bool,
) -> dict:
    """blocks are page_parser Khối in reading order: {"bbox", "label" (Nhãn Monkey),
    "content" (raw)}. Picture crops come from the full-resolution page here.

    page_count is the PDF page count, or None for image input, as in PaddleOCR-VL.

    As in PaddleOCR-VL, model_settings["markdown_ignore_labels"] drops Khối from
    the Markdown only; they stay in parsing_res_list.

    Picture crops go to markdown.images as Base64 JPEG under a Paddle-style path
    that the Markdown references. markdown.images holds only referenced crops; with
    return_markdown_images False it is None and the Markdown has no image reference.
    """
    width, height = page.size
    markdown_ignore_labels = model_settings["markdown_ignore_labels"]
    parsing_res_list = []
    fragments = []
    images = {} if return_markdown_images else None
    for index, block in enumerate(blocks):
        label = PADDLE_LABELS.get(block["label"], block["label"])
        content, markdown = _format_block(label, block["content"])
        if block["label"] == "Picture" and images is not None and label not in markdown_ignore_labels:
            x1, y1, x2, y2 = block["bbox"]
            # WeKnora merges page image maps, so bbox-only paths would collide.
            path = f"imgs/page_{page_index + 1}/img_in_{label}_box_{x1}_{y1}_{x2}_{y2}.jpg"
            with page.crop(tuple(block["bbox"])) as crop:
                images[path] = _jpeg_base64(crop)
            markdown = _image_markdown(path, block["bbox"], width)
        parsing_res_list.append({
            "block_label": label,
            "monkey_label": block["label"],
            "block_content": content,
            "block_bbox": block["bbox"],
            "block_id": index,
            "block_order": index + 1,
        })
        if markdown and label not in markdown_ignore_labels:
            fragments.append(markdown)
    return {
        "prunedResult": {
            "page_count": page_count,
            "width": width,
            "height": height,
            "model_settings": model_settings,
            "parsing_res_list": parsing_res_list,
        },
        "markdown": {
            # Upstream result2md drops U+FFFD from the Markdown.
            "text": "\n\n".join(fragments).replace("\ufffd", ""),
            "images": images,
        },
        "outputImages": None,
        "inputImage": None,
    }
