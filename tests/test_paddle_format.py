"""Paddle-style Khối formatting, shared by both parsing modes.

Nhãn Paddle mapping, HTML tables, LaTeX formulas, Paddle-style Markdown and
markdownIgnoreLabels, checked through the HTTP API against fake vLLM.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO

import pytest
from PIL import Image, ImageDraw

from conftest import (
    END2END_PROMPT,
    FORMULA_PROMPT,
    LAYOUT_PROMPT,
    TABLE_PROMPT,
    TEXT_PROMPT,
    image_base64,
)
from fake_vllm import FakeVLLM

# 1000 x 1000 pixels, so 0-1000 model coordinates are page pixels.
PAGE_SIZE = (1000, 1000)

# Synthetic Khối with every label that has a two-stage prompt, plus Picture.
# "raw" is what the model returns, in both modes.
SHARED = [
    {"label": "Title", "bbox": [0, 0, 1000, 50], "color": (200, 0, 0),
     "raw": "Quy chế mẫu\nvề bảo mật dữ liệu"},
    {"label": "Section-header", "bbox": [0, 60, 1000, 100], "color": (0, 200, 0),
     "raw": "Chương I. Quy định chung"},
    {"label": "Text", "bbox": [0, 110, 1000, 200], "color": (0, 0, 200),
     "raw": "Điều 1. Văn bản tổng hợp\ufffd dùng để kiểm thử."},
    {"label": "List-item", "bbox": [0, 210, 1000, 250], "color": (200, 200, 0),
     "raw": "a) Mục thứ nhất"},
    {"label": "Formula", "bbox": [100, 260, 900, 320], "color": (0, 200, 200),
     "raw": "$$E = m c^{2} \\quad (1)$$"},
    {"label": "Table", "bbox": [0, 330, 1000, 500], "color": (200, 0, 200),
     "raw": "<fcel>Mục<lcel><nl><fcel>A&B<fcel>2<nl>"},
    {"label": "Caption", "bbox": [0, 510, 1000, 540], "color": (90, 90, 90),
     "raw": "Bảng 1. Ví dụ"},
    {"label": "Picture", "bbox": [300, 550, 700, 700], "color": (150, 60, 0), "raw": ""},
    {"label": "Page-header", "bbox": [0, 900, 1000, 930], "color": (0, 60, 150),
     "raw": "Văn bản mẫu"},
    {"label": "Page-footer", "bbox": [0, 950, 1000, 990], "color": (60, 150, 0),
     "raw": "Trang 1"},
]
# Labels without a two-stage prompt (empty there), so only end-to-end returns content.
EXTRA = [
    {"label": "Footnote", "bbox": [0, 800, 1000, 840], "raw": "(1) Chú thích mẫu."},
    {"label": "Checkbox", "bbox": [0, 850, 100, 880], "raw": "x"},  # not in the mapping
]

# PaddleOCR-VL serving default (prettifyMarkdown=true): PaddleX format_image_scaled_by_html
# wrapped by format_centered_by_html; width is the bbox share of the page width.
PICTURE_PATH = "imgs/img_in_image_box_300_550_700_700.jpg"
PICTURE_MARKDOWN = f'<div style="text-align: center;"><img src="{PICTURE_PATH}" alt="Image" width="40%" /></div>'
TABLE_HTML = '<table><tr><td colspan="2">Mục</td></tr><tr><td>A&amp;B</td><td>2</td></tr></table>'
EXPECTED = {
    # monkey_label: (block_label, block_content, Markdown fragment)
    "Title": ("doc_title", "Quy chế mẫu\nvề bảo mật dữ liệu", "# Quy chế mẫu về bảo mật dữ liệu"),
    "Section-header": ("paragraph_title", "Chương I. Quy định chung", "## Chương I. Quy định chung"),
    "Text": ("text", "Điều 1. Văn bản tổng hợp\ufffd dùng để kiểm thử.",
             "Điều 1. Văn bản tổng hợp dùng để kiểm thử."),
    "List-item": ("text", "a) Mục thứ nhất", "a) Mục thứ nhất"),
    "Formula": ("formula", "E = m c^{2}", "$$\nE = m c^{2}\n$$\n1"),
    "Table": ("table", TABLE_HTML, TABLE_HTML),
    "Caption": ("figure_title", "Bảng 1. Ví dụ", "Bảng 1. Ví dụ"),
    "Picture": ("image", "", PICTURE_MARKDOWN),
    "Page-header": ("header", "Văn bản mẫu", "Văn bản mẫu"),
    "Page-footer": ("footer", "Trang 1", "Trang 1"),
    "Footnote": ("footnote", "(1) Chú thích mẫu.", "(1) Chú thích mẫu."),
    "Checkbox": ("Checkbox", "x", "x"),
}
PADDLE_DEFAULT_IGNORE = ["number", "footnote", "header", "header_image", "footer", "footer_image", "aside_text"]


def _model_output(blocks, with_content: bool) -> str:
    keys = ("bbox", "label", "content") if with_content else ("bbox", "label")
    return json.dumps(
        [{k: v for k, v in {**b, "content": b["raw"]}.items() if k in keys} for b in blocks],
        ensure_ascii=False,
    )


def _painted_page() -> Image.Image:
    page = Image.new("RGB", PAGE_SIZE, "white")
    draw = ImageDraw.Draw(page)
    for block in SHARED:
        x1, y1, x2, y2 = block["bbox"]
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=block["color"])
    return page


def _recognize(image: Image.Image) -> str:
    color = image.convert("RGB").getpixel((0, 0))
    return next(b["raw"] for b in SHARED if b["color"] == color) + "\n"


@pytest.fixture
def fake_vllm(monkeypatch):
    answers = {
        END2END_PROMPT: _model_output(SHARED + EXTRA, with_content=True),
        LAYOUT_PROMPT: _model_output(SHARED, with_content=False),
        TEXT_PROMPT: _recognize,
        FORMULA_PROMPT: _recognize,
        TABLE_PROMPT: _recognize,
    }
    with FakeVLLM(answers) as fake:
        monkeypatch.setenv("MOCR2_SERVER_URL", fake.url)
        monkeypatch.setenv("MOCR2_SERVED_MODEL_NAME", "fake-monkey")
        yield fake


def _parse(gateway, mode="end-to-end", **fields) -> dict:
    response = gateway.post(
        "/layout-parsing",
        json={
            "file": image_base64(_painted_page()),
            "fileType": 1,
            "vlmExtraArgs": {"parsingMode": mode},
            **fields,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["result"]["layoutParsingResults"][0]


def _markdown(labels, ignore) -> str:
    fragments = [
        EXPECTED[label][2]
        for label in labels
        if EXPECTED[label][2] is not None and EXPECTED[label][0] not in ignore
    ]
    return "\n\n".join(fragments)


ALL_LABELS = [b["label"] for b in SHARED + EXTRA]


def test_monkey_labels_map_to_paddle_labels_and_keep_monkey_label(gateway):
    items = _parse(gateway)["prunedResult"]["parsing_res_list"]

    assert [(item["monkey_label"], item["block_label"]) for item in items] == [
        (label, EXPECTED[label][0]) for label in ALL_LABELS
    ]


def test_block_content_is_plain_text_html_table_and_latex_formula(gateway):
    items = _parse(gateway)["prunedResult"]["parsing_res_list"]

    assert {item["monkey_label"]: item["block_content"] for item in items} == {
        label: expected[1] for label, expected in EXPECTED.items()
    }


@pytest.mark.parametrize("fields", [{}, {"markdownIgnoreLabels": None}], ids=["omitted", "null"])
def test_markdown_is_paddle_style_and_drops_paddle_default_labels(gateway, fields):
    result = _parse(gateway, **fields)

    assert result["markdown"]["text"] == _markdown(ALL_LABELS, PADDLE_DEFAULT_IGNORE)
    assert "\ufffd" not in result["markdown"]["text"]
    assert result["prunedResult"]["model_settings"]["markdown_ignore_labels"] == PADDLE_DEFAULT_IGNORE
    # Ignored Khối stay in parsing_res_list.
    assert len(result["prunedResult"]["parsing_res_list"]) == len(ALL_LABELS)


@pytest.mark.parametrize("ignore", [["table", "formula"], []], ids=["explicit", "empty"])
def test_explicit_markdown_ignore_labels_are_honoured(gateway, ignore):
    result = _parse(gateway, markdownIgnoreLabels=ignore)

    assert result["markdown"]["text"] == _markdown(ALL_LABELS, ignore)
    assert result["prunedResult"]["model_settings"]["markdown_ignore_labels"] == ignore
    assert len(result["prunedResult"]["parsing_res_list"]) == len(ALL_LABELS)


def test_markdown_ignore_labels_use_paddle_labels_not_monkey_labels(gateway):
    result = _parse(gateway, markdownIgnoreLabels=["Text", "doc_title"])

    markdown = result["markdown"]["text"]
    assert not markdown.startswith("# ")
    assert "Điều 1." in markdown


def test_both_modes_give_the_same_format_for_the_same_khối(gateway):
    shared_labels = [b["label"] for b in SHARED]
    end_to_end = _parse(gateway, "end-to-end", markdownIgnoreLabels=[])
    two_stage = _parse(gateway, "two-stage", markdownIgnoreLabels=[])

    # End-to-end also returns the EXTRA Khối, which come last.
    e2e_items = end_to_end["prunedResult"]["parsing_res_list"][: len(SHARED)]
    assert two_stage["prunedResult"]["parsing_res_list"] == e2e_items
    assert [item["monkey_label"] for item in e2e_items] == shared_labels
    assert two_stage["markdown"]["text"] == _markdown(shared_labels, [])
    assert two_stage["markdown"]["images"] == end_to_end["markdown"]["images"]
    assert end_to_end["markdown"]["text"].startswith(two_stage["markdown"]["text"] + "\n\n")


MODES = ["end-to-end", "two-stage"]


@pytest.mark.parametrize("mode", MODES)
def test_picture_is_cropped_into_markdown_images_and_referenced_from_markdown(gateway, mode):
    result = _parse(gateway, mode)

    images = result["markdown"]["images"]
    assert list(images) == [PICTURE_PATH]
    assert PICTURE_MARKDOWN in result["markdown"]["text"]
    crop = Image.open(BytesIO(base64.b64decode(images[PICTURE_PATH])))
    assert crop.format == "JPEG"
    assert crop.size == (400, 150)  # the Picture bbox in page pixels
    picture_color = next(b["color"] for b in SHARED if b["label"] == "Picture")
    for x, y in [(0, 0), (399, 149), (200, 75)]:
        assert all(abs(a - b) <= 8 for a, b in zip(crop.convert("RGB").getpixel((x, y)), picture_color))


@pytest.mark.parametrize("mode", MODES)
def test_return_markdown_images_false_returns_no_images_and_no_dangling_reference(gateway, mode):
    result = _parse(gateway, mode, returnMarkdownImages=False)

    assert result["markdown"].get("images") is None
    assert "imgs/" not in result["markdown"]["text"]
    picture = next(i for i in result["prunedResult"]["parsing_res_list"] if i["monkey_label"] == "Picture")
    assert picture["block_content"] == ""


@pytest.mark.parametrize("mode", MODES)
def test_ignored_picture_is_left_out_of_markdown_and_images(gateway, mode):
    result = _parse(gateway, mode, markdownIgnoreLabels=["image"])

    assert "imgs/" not in result["markdown"]["text"]
    assert result["markdown"]["images"] == {}
    labels = [i["block_label"] for i in result["prunedResult"]["parsing_res_list"]]
    assert "image" in labels
