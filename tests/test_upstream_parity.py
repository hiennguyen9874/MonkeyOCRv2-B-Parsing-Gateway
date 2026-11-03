"""Parity with Upstream MonkeyOCRv2 helpers at revision 106914e.

Cases are copied from references/MonkeyOCRv2/parsing/tests/test_model_output_parsing.py,
test_bbox.py, test_otsl_to_html.py and the layout half of test_fake_model_pipeline.py /
test_fake_server_pipeline.py so that drift shows up when the pinned revision changes.
The load_image and process_formula cases have no Upstream test; their expected
values are worked out by hand from Upstream's code. otsl_to_html and
process_formula are also run against the functions in Upstream's own source.
"""

from __future__ import annotations

import ast
import random
import re
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from PIL import Image, ImageDraw

from monkeyocr_gateway.document_loader import PDFIUM_LOCK, PdfDocument

from monkeyocr_gateway.upstream import (
    ALL_PROMPT,
    _build_page_tasks,
    _map_bbox_to_image,
    _should_retry_repeat_output,
    detect_repeat_token,
    load_image,
    otsl_to_html,
    parse_end2end_output,
    parse_layout_output,
    process_formula,
)

CORE_RUNNER = Path(__file__).parents[1] / "references/MonkeyOCRv2/parsing/core_runner.py"


def _labels(records):
    return [item["label"] for item in records]


def _upstream_source():
    if not CORE_RUNNER.exists():
        pytest.skip("references/MonkeyOCRv2 is not checked out (git submodule update --init)")
    return ast.parse(CORE_RUNNER.read_text(encoding="utf-8"))


def _upstream_function(name: str):
    """Compile one helper from Upstream's source; importing core_runner would pull in torch."""
    node = next(n for n in _upstream_source().body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"re": re, "Image": Image, "PDFIUM_LOCK": PDFIUM_LOCK}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(CORE_RUNNER), "exec"), namespace)
    return namespace[name]


# --- _render_pdf_page at 106914e ---------------------------------------------


def test_pdf_rendering_matches_upstream():
    render = _upstream_function("_render_pdf_page")
    images = []
    for size, color in [((173, 251), "white"), ((317, 149), "yellow"), ((121, 193), "cyan")]:
        image = Image.new("RGB", size, color)
        draw = ImageDraw.Draw(image)
        draw.rectangle((13, 19, 81, 103), fill="red")
        draw.line((0, 0, size[0] - 1, size[1] - 1), fill="blue", width=3)
        images.append(image)
    buffer = BytesIO()
    images[0].save(buffer, "PDF", save_all=True, append_images=images[1:], resolution=144)
    data = buffer.getvalue()
    document = PdfDocument(data)
    try:
        with PDFIUM_LOCK, pdfium.PdfDocument(data) as upstream_pdf:
            for index in range(len(images)):
                expected = render(upstream_pdf, index)
                actual = document.page(index)
                try:
                    assert actual.mode == expected.mode == "RGB"
                    assert actual.size == expected.size
                    assert actual.tobytes() == expected.tobytes()
                finally:
                    actual.close()
                    expected.close()
    finally:
        document.close()


# --- test_model_output_parsing.py -------------------------------------------


def test_valid_json_list():
    text = (
        '[{"bbox": [0, 0, 100, 100], "label": "Text", "content": "hello"},'
        ' {"bbox": [200, 200, 400, 400], "label": "Title", "content": "Hi"}]'
    )
    records, layouts = parse_end2end_output(text, (1000, 1000))
    assert _labels(records) == ["Text", "Title"]
    assert records[0]["content"] == "hello"
    assert records[0]["bbox"] == [0, 0, 100, 100]
    assert layouts[1]["label"] == "Title"
    assert "content" not in layouts[1]


def test_valid_json_dict():
    text = '{"bbox": [0, 0, 100, 100], "label": "Title", "content": "Hi"}'
    records, layouts = parse_end2end_output(text, (1000, 1000))
    assert len(records) == 1
    assert records[0]["label"] == "Title"
    assert records[0]["content"] == "Hi"
    assert layouts[0]["bbox"] == [0, 0, 100, 100]


def test_python_literal_list():
    text = "[{'bbox': [0, 0, 50, 50], 'label': 'Text', 'content': 'py'}]"
    records, _layouts = parse_end2end_output(text, (100, 100))
    assert records[0]["content"] == "py"
    assert records[0]["bbox"] == [0, 0, 5, 5]


def test_truncated_json_keeps_complete_items():
    text = (
        '[{"bbox": [0, 0, 100, 100], "label": "Text", "content": "ok"},'
        ' {"bbox": [10, 10'
    )
    records, layouts = parse_end2end_output(text, (1000, 1000))
    assert len(records) == 1
    assert records[0]["content"] == "ok"
    assert layouts[0]["label"] == "Text"


def test_truncated_single_object_still_parses():
    text = '[{ "bbox": [1, 2, 3, 4], "label": "Text" '
    records, _layouts = parse_end2end_output(text, (1000, 1000))
    assert len(records) == 1
    assert records[0]["label"] == "Text"


def test_mixed_non_json_prose_around_list():
    text = (
        "Here is the layout:\n"
        '[{"bbox": [0, 0, 500, 80], "label": "Title", "content": "Hello"}]\n'
        "Thanks."
    )
    records, _layouts = parse_end2end_output(text, (200, 100))
    assert len(records) == 1
    assert records[0]["content"] == "Hello"
    assert records[0]["bbox"] == [0, 0, 100, 8]


def test_empty_and_non_structured_text_return_empty():
    assert parse_end2end_output("", (10, 10)) == ([], [])
    assert parse_end2end_output("   ", (10, 10)) == ([], [])
    assert parse_end2end_output("not json at all", (10, 10)) == ([], [])


def test_invalid_items_are_skipped():
    text = (
        '[{"bbox": [1, 2], "label": "Text"},'
        ' {"bbox": [1, 2, 3, 4]},'
        ' {"bbox": [0, 0, 10, 10], "label": "Title", "content": "keep"}]'
    )
    records, _layouts = parse_end2end_output(text, (100, 100))
    assert _labels(records) == ["Title"]
    assert records[0]["content"] == "keep"


# --- ALL_PROMPT --------------------------------------------------------------


def test_all_prompt_matches_upstream_source():
    # Read the literal from Upstream's source; importing it would pull in torch.
    tree = _upstream_source()
    upstream = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ALL_PROMPT" for target in node.targets)
    )
    assert ALL_PROMPT == upstream


# --- test_fake_model_pipeline.py / test_fake_server_pipeline.py (get_layout) --


def test_layout_output_maps_to_pixel_bboxes_without_content():
    text = '[{"bbox": [0, 0, 1000, 1000], "label": "Text"}]'
    assert parse_layout_output(text, (32, 16)) == [{"bbox": [0, 0, 32, 16], "label": "Text"}]
    assert parse_layout_output(text, (40, 20)) == [{"bbox": [0, 0, 40, 20], "label": "Text"}]


def test_layout_output_is_tolerant_like_end2end():
    text = '[{"bbox": [0, 0, 100, 100], "label": "Text", "content": "x"}, {"bbox": [10, 10'
    assert parse_layout_output(text, (1000, 1000)) == [{"bbox": [0, 0, 100, 100], "label": "Text"}]


def test_fake_model_layout_to_tasks():
    image = Image.new("RGB", (32, 16), "white")
    tasks = _build_page_tasks(0, image, parse_layout_output('[{"bbox": [0, 0, 1000, 1000], "label": "Text"}]', image.size))
    assert len(tasks) == 1
    assert tasks[0]["question"] == ALL_PROMPT["Text"]
    assert tasks[0]["need_infer"] is True
    assert tasks[0]["image"].size == (32, 16)


def test_labels_missing_from_prompt_map_need_no_inference():
    image = Image.new("RGB", (10, 10))
    tasks = _build_page_tasks(
        0,
        image,
        [{"bbox": [0, 0, 5, 5], "label": "Footnote"}, {"bbox": [0, 0, 5, 5], "label": "Picture"}],
    )
    assert [(task["need_infer"], task["question"]) for task in tasks] == [(False, ""), (False, "")]
    assert [task["block_idx"] for task in tasks] == [0, 1]


# --- test_bbox.py -------------------------------------------------------------


def test_full_frame_maps_to_image_size():
    assert _map_bbox_to_image([0, 0, 1000, 1000], 100, 200) == [0, 0, 100, 200]


def test_out_of_range_coordinates_are_clamped():
    assert _map_bbox_to_image([-50, -50, 2000, 2000], 100, 200) == [0, 0, 100, 200]


def test_negative_coordinates_clamp_to_origin_with_nonzero_area():
    assert _map_bbox_to_image([-10, 10, 20, -20], 50, 50) == [0, 0, 1, 1]


def test_zero_area_box_is_expanded_to_one_pixel():
    assert _map_bbox_to_image([500, 500, 500, 500], 100, 100) == [50, 50, 51, 51]


def test_swapped_corners_are_ordered():
    assert _map_bbox_to_image([800, 200, 100, 900], 100, 100) == [10, 20, 80, 90]


def test_build_page_tasks_clamps_pixel_bboxes():
    image = Image.new("RGB", (10, 10), color=(0, 0, 0))
    tasks = _build_page_tasks(
        0,
        image,
        [
            {"bbox": [-10, -10, 100, 100], "label": "Text"},
            {"bbox": [5, 5, 5, 5], "label": "Title"},
        ],
    )
    assert tasks[0]["bbox"] == [0, 0, 10, 10]
    assert tasks[1]["bbox"] == [5, 5, 6, 6]
    assert tasks[0]["image"].size == (10, 10)
    assert tasks[1]["image"].size == (1, 1)


# --- test_otsl_to_html.py --------------------------------------------------


class _TableParser(HTMLParser):
    """Upstream tests/html_table.py: collects tags and cell texts."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []
        self.cell_texts: list[str] = []
        self._in_td = False
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == "td":
            self._in_td = True
            self._buf = []

    def handle_endtag(self, tag):
        if tag == "td" and self._in_td:
            self.cell_texts.append("".join(self._buf))
            self._in_td = False
            self._buf = []

    def handle_data(self, data):
        if self._in_td:
            self._buf.append(data)


def _parse_table(html: str) -> _TableParser:
    assert html.startswith("<table>"), html
    assert html.endswith("</table>"), html
    parser = _TableParser()
    parser.feed(html)
    parser.close()
    assert parser.tags[0] == "table"
    return parser


def test_empty_otsl_returns_empty_table():
    assert otsl_to_html("") == "<table></table>"
    assert otsl_to_html("   ") == "<table></table>"
    assert otsl_to_html(None) == "<table></table>"


def test_basic_2x2_table():
    html = otsl_to_html("<fcel>A<fcel>B<nl><fcel>C<fcel>D<nl>")
    assert html == (
        "<table><tr><td>A</td><td>B</td></tr>"
        "<tr><td>C</td><td>D</td></tr></table>"
    )
    assert _parse_table(html).cell_texts == ["A", "B", "C", "D"]


def test_row_split_across_newlines_issue_24():
    html = otsl_to_html("<fcel>line1\nline2<fcel>B<nl>")
    assert html == "<table><tr><td>line1<br>line2</td><td>B</td></tr></table>"
    _parse_table(html)


def test_crlf_in_cell_becomes_br():
    html = otsl_to_html("<fcel>line1\r\nline2<nl>")
    assert html == "<table><tr><td>line1<br>line2</td></tr></table>"


def test_colspan_from_lcel_tokens():
    html = otsl_to_html("<fcel>Head<lcel><lcel><nl><fcel>A<fcel>B<fcel>C<nl>")
    assert html == (
        '<table><tr><td colspan="3">Head</td></tr>'
        "<tr><td>A</td><td>B</td><td>C</td></tr></table>"
    )


def test_rowspan_from_ucel_tokens():
    html = otsl_to_html("<fcel>Side<fcel>A<nl><ucel><fcel>B<nl>")
    assert html == (
        '<table><tr><td rowspan="2">Side</td><td>A</td></tr>'
        "<tr><td>B</td></tr></table>"
    )


def test_rowspan_and_colspan_with_xcel():
    html = otsl_to_html("<fcel>Big<lcel><fcel>A<nl><ucel><xcel><fcel>B<nl>")
    assert html == (
        '<table><tr><td rowspan="2" colspan="2">Big</td><td>A</td></tr>'
        "<tr><td>B</td></tr></table>"
    )


def test_empty_cell_token():
    html = otsl_to_html("<fcel>A<ecel><fcel>B<nl>")
    assert html == "<table><tr><td>A</td><td></td><td>B</td></tr></table>"


MALFORMED_OTSL = [
    "<nl>",
    "<fcel>A",
    "<zzzz>nope<fcel>ok<nl>",
    "notags just text",
    "<fcel>A<lcel><ucel><xcel><nl>",
    "<fcel>",
    "<<<>>>",
    "<fcel>A<nl><nl><fcel>B<nl>",
]


@pytest.mark.parametrize("otsl", MALFORMED_OTSL)
def test_malformed_otsl_does_not_raise_and_returns_table(otsl):
    _parse_table(otsl_to_html(otsl))


def test_html_special_characters_are_escaped():
    html = otsl_to_html("<fcel>A&B<\"'><nl>")
    assert html == "<table><tr><td>A&amp;B&lt;&quot;&#x27;&gt;</td></tr></table>"


OTSL_CASES = MALFORMED_OTSL + [
    "",
    "<fcel>A&B<\"'><nl>",
    "<fcel>Big<lcel><fcel>A<nl><ucel><xcel><fcel>B<nl>",
    "<fcel>a<br/>b &amp; <b>c</b><fcel>\ue100fcel>lit<nl>",
    "<otsl><fcel>A<nl></otsl> trailing <otsl><fcel>B<nl></otsl>",
]


@pytest.mark.parametrize("otsl", OTSL_CASES)
def test_otsl_to_html_matches_upstream_source(otsl):
    assert otsl_to_html(otsl) == _upstream_function("otsl_to_html")(otsl)


# --- process_formula ----------------------------------------------------------

FORMULA_CASES = [
    ("$$x^2$$", ("x^2", None)),
    ("E = m c^{2} \\quad (1)", ("E = m c^{2}", "1")),
    # Upstream reads only group(1), so a \tag number is dropped, not returned.
    ("a + b \\tag{3}", ("a + b", None)),
    ("x \\quad \\quad \\quad \\quad \\quad y", ("x \\quad y", None)),
    (
        "\\begin{aligned} a &= b \\\\ c &= d \\end{aligned} \\qquad (2)",
        ("\\begin{aligned}\na &= b \\\\ c &= d\n\\end{aligned}", "2"),
    ),
    ("\\begin{cases} x \\eqno (4) \\end{cases}", ("\\begin{cases}\nx\n\\end{cases}", "4")),
    ("", ("", None)),
]


@pytest.mark.parametrize(("raw", "expected"), FORMULA_CASES)
def test_process_formula(raw, expected):
    assert process_formula(raw) == expected


@pytest.mark.parametrize(("raw", "_expected"), FORMULA_CASES)
def test_process_formula_matches_upstream_source(raw, _expected):
    assert process_formula(raw) == _upstream_function("process_formula")(raw)


# --- load_image request resizing --------------------------------------------


def test_image_above_max_pixels_is_downscaled():
    image = Image.new("RGB", (1240, 1754), "white")
    assert load_image(image, max_pixels=1003520).size == (842, 1191)


def test_image_within_max_pixels_keeps_its_size():
    image = Image.new("RGB", (800, 600), "white")
    assert load_image(image, max_pixels=1003520).size == (800, 600)


def test_image_below_min_pixels_is_upscaled():
    # scale = sqrt(1003520 / 40000) = 5.0089...
    image = Image.new("RGB", (200, 200), "white")
    assert load_image(image, min_pixels=1003520).size == (1001, 1001)


def test_extreme_aspect_ratio_becomes_dummy_image():
    image = Image.new("RGB", (2010, 10), "white")
    assert load_image(image).size == (32, 32)


def test_result_is_rgb():
    image = Image.new("L", (10, 10), 255)
    assert load_image(image).mode == "RGB"


# --- test_repeat_detection.py -------------------------------------------------


def _reference_detect_repeat_token(
    predicted_tokens: str,
    base_max_repeats: int = 4,
    window_size: int = 500,
    cut_from_end: int = 0,
    scaling_factor: float = 3.0,
) -> bool:
    """Upstream's reference implementation of the original backward-scan algorithm."""
    if cut_from_end > 0:
        predicted_tokens = predicted_tokens[:-cut_from_end]
    for seq_len in range(1, window_size // 2 + 1):
        candidate_seq = predicted_tokens[-seq_len:]
        max_repeats = int(base_max_repeats * (1 + scaling_factor / seq_len))
        repeat_count = 0
        pos = len(predicted_tokens) - seq_len
        if pos < 0:
            continue
        while pos >= 0:
            if predicted_tokens[pos : pos + seq_len] == candidate_seq:
                repeat_count += 1
                pos -= seq_len
            else:
                break
        if repeat_count > max_repeats:
            return True
    return False


@pytest.mark.parametrize(
    "value",
    ["", "abc", "A" * 20, "AB" * 20, "prefix-" + "xyz" * 30, "A" * 19 + "B"],
)
def test_repeat_detector_matches_reference(value):
    assert detect_repeat_token(value) == _reference_detect_repeat_token(value)


def test_repeat_detector_matches_reference_for_random_inputs():
    rng = random.Random(0)
    alphabet = "abcXYZ012"
    for _ in range(200):
        value = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 200)))
        cut = rng.randrange(0, 60)
        window = rng.randrange(2, 80)
        assert detect_repeat_token(value, window_size=window, cut_from_end=cut) == (
            _reference_detect_repeat_token(value, window_size=window, cut_from_end=cut)
        )


@pytest.mark.parametrize(
    "raw",
    [None, "", "Điều 1. Văn bản tổng hợp.", "ha" * 40, "ha" * 40 + "x" * 51, "ab" * 10 + "c" * 51],
)
def test_should_retry_repeat_output_matches_upstream_source(raw):
    upstream = _upstream_function("_should_retry_repeat_output")
    upstream.__globals__["detect_repeat_token"] = _upstream_function("detect_repeat_token")
    assert _should_retry_repeat_output(raw) == upstream(raw)
