"""Build PaddleOCR-VL Kết quả trang from Khối.

Both parsing modes go through this module, so the same Khối gives the same output.
"""

from __future__ import annotations

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
        # Crops in markdown.images come with ticket #7; no OCR of image content.
        return "", ""
    if not raw:
        return "", ""
    if label == "doc_title":
        return raw, "# " + _collapse_soft_newlines(raw)
    if label == "paragraph_title":
        return raw, "## " + _collapse_soft_newlines(raw)
    return raw, raw


def build_page_result(
    width: int, height: int, blocks: list[dict], model_settings: dict, page_count: int | None
) -> dict:
    """blocks are page_parser Khối in reading order: {"bbox", "label" (Nhãn Monkey), "content" (raw)}.

    page_count is the PDF page count, or None for image input, as in PaddleOCR-VL.

    As in PaddleOCR-VL, model_settings["markdown_ignore_labels"] drops Khối from
    the Markdown only; they stay in parsing_res_list.
    """
    markdown_ignore_labels = model_settings["markdown_ignore_labels"]
    parsing_res_list = []
    fragments = []
    for index, block in enumerate(blocks):
        label = PADDLE_LABELS.get(block["label"], block["label"])
        content, markdown = _format_block(label, block["content"])
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
            "images": {},
        },
        "outputImages": None,
        "inputImage": None,
    }
