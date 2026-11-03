"""Parity with Upstream MonkeyOCRv2 helpers at revision 106914e.

Cases are copied from references/MonkeyOCRv2/parsing/tests/test_model_output_parsing.py,
test_bbox.py and the layout half of test_fake_model_pipeline.py /
test_fake_server_pipeline.py so that drift shows up when the pinned revision changes.
The load_image cases have no Upstream test; their expected sizes are worked out
by hand from Upstream's formula.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from PIL import Image

from monkeyocr_gateway.upstream import (
    ALL_PROMPT,
    _build_page_tasks,
    _map_bbox_to_image,
    load_image,
    parse_end2end_output,
    parse_layout_output,
)

CORE_RUNNER = Path(__file__).parents[1] / "references/MonkeyOCRv2/parsing/core_runner.py"


def _labels(records):
    return [item["label"] for item in records]


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
    if not CORE_RUNNER.exists():
        pytest.skip("references/MonkeyOCRv2 is not checked out (git submodule update --init)")
    tree = ast.parse(CORE_RUNNER.read_text(encoding="utf-8"))
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
