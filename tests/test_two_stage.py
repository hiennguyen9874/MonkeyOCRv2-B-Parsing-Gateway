"""Chế độ hai giai đoạn and choosing the mode with vlmExtraArgs.parsingMode."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageChops, ImageDraw

from conftest import (
    END2END_PROMPT,
    FORMULA_PROMPT,
    LAYOUT_PROMPT,
    TABLE_PROMPT,
    TEXT_PROMPT,
    image_base64,
    recorded,
)
from fake_vllm import FakeVLLM, request_image, request_prompt
from monkeyocr_gateway.app import create_app

PAGE_SIZE = (1240, 1754)

# Khối of layout_synthetic_page.txt in layout order, with their bboxes worked out
# by hand from 0-1000 model space into PAGE_SIZE pixels. Each Khối is painted a
# unique colour so the fake model can tell which crop it received.
BLOCKS = [
    {"label": "Title", "bbox": [124, 88, 1116, 210], "color": (200, 0, 0)},
    {"label": "Text", "bbox": [124, 263, 1116, 702], "color": (0, 200, 0)},
    {"label": "Formula", "bbox": [248, 737, 992, 842], "color": (0, 0, 200)},
    {"label": "Table", "bbox": [124, 877, 1116, 1403], "color": (200, 200, 0)},
    {"label": "Picture", "bbox": [372, 1421, 868, 1473], "color": (0, 200, 200)},
    {"label": "Footnote", "bbox": [124, 1491, 1116, 1579], "color": (200, 0, 200)},
    {"label": "Page-footer", "bbox": [99, 1666, 1141, 1719], "color": (90, 90, 90)},
]
BY_COLOR = {block["color"]: block for block in BLOCKS}
PROMPT_BY_LABEL = {
    "Title": TEXT_PROMPT,
    "Text": TEXT_PROMPT,
    "Formula": FORMULA_PROMPT,
    "Table": TABLE_PROMPT,
    "Page-footer": TEXT_PROMPT,
}
CONTENT_BY_LABEL = {
    "Title": "Quy chế mẫu về bảo mật dữ liệu",
    "Text": "Điều 1. Văn bản này là dữ liệu tổng hợp.",
    "Formula": "E = m c^{2}",
    "Table": "<fcel>A<fcel>B<nl><fcel>1<fcel>2<nl>",
    "Page-footer": "Trang 1",
}

def _painted_page(size=PAGE_SIZE) -> Image.Image:
    page = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(page)
    for block in BLOCKS:
        x1, y1, x2, y2 = block["bbox"]
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=block["color"])
    return page


def _khối_of(image: Image.Image) -> dict:
    return BY_COLOR[image.convert("RGB").getpixel((0, 0))]


def _recognize(image: Image.Image) -> str:
    block = BY_COLOR.get(image.convert("RGB").getpixel((0, 0)))
    if block is None:  # crop of an unpainted page
        return ""
    label = block["label"]
    # Trailing whitespace checks that raw output is stripped, as Upstream does.
    return CONTENT_BY_LABEL[label] + "\n"


@pytest.fixture
def fake_vllm(monkeypatch):
    answers = {
        LAYOUT_PROMPT: recorded("layout_synthetic_page.txt"),
        END2END_PROMPT: recorded("end2end_synthetic_page.txt"),
        TEXT_PROMPT: _recognize,
        FORMULA_PROMPT: _recognize,
        TABLE_PROMPT: _recognize,
    }
    with FakeVLLM(answers) as fake:
        monkeypatch.setenv("MOCR2_SERVER_URL", fake.url)
        monkeypatch.setenv("MOCR2_SERVED_MODEL_NAME", "fake-monkey")
        monkeypatch.delenv("MOCR2_PARSING_MODE", raising=False)
        yield fake


def _post(client, page=None, **extra):
    page = page or _painted_page()
    return client.post("/layout-parsing", json={"file": image_base64(page), "fileType": 1, **extra})


def _first_page(response) -> dict:
    assert response.status_code == 200, response.text
    return response.json()["result"]["layoutParsingResults"][0]


def test_two_stage_sends_one_layout_request_then_one_recognition_per_prompted_khối(gateway, fake_vllm):
    page = _painted_page()
    _first_page(_post(gateway, page, vlmExtraArgs={"parsingMode": "two-stage"}))

    layout, *recognitions = fake_vllm.requests
    assert request_prompt(layout["payload"]) == LAYOUT_PROMPT
    assert layout["payload"]["max_tokens"] == 4096
    assert request_image(layout["payload"]).size == (842, 1191)  # page resized to the pixel limit

    received = {}
    for request in recognitions:
        payload = request["payload"]
        assert payload["max_tokens"] == 5000
        assert payload["temperature"] == 0
        crop = request_image(payload)
        block = _khối_of(crop)
        assert request_prompt(payload) == PROMPT_BY_LABEL[block["label"]]
        # Cropped from the full-resolution page, not from the resized layout image.
        expected = page.crop(block["bbox"])
        assert crop.size == expected.size
        assert ImageChops.difference(crop.convert("RGB"), expected).getbbox() is None
        received[block["label"]] = received.get(block["label"], 0) + 1
    assert received == {"Title": 1, "Text": 1, "Formula": 1, "Table": 1, "Page-footer": 1}


def test_parsing_res_list_keeps_layout_order_when_recognition_finishes_out_of_order(gateway, fake_vllm):
    def finished_labels():
        return [_khối_of(request_image(p))["label"] for p in fake_vllm.completed[1:]]

    def recognize_out_of_order(image):
        label = _khối_of(image)["label"]
        prerequisites = {
            "Text": {"Formula", "Table", "Page-footer"},
            "Title": {"Text"},
        }.get(label, set())
        with fake_vllm.completed_condition:
            assert fake_vllm.completed_condition.wait_for(
                lambda: prerequisites.issubset(finished_labels()), timeout=5
            ), f"Recognition stalled waiting for {prerequisites}"
        return _recognize(image)

    for prompt in (TEXT_PROMPT, FORMULA_PROMPT, TABLE_PROMPT):
        fake_vllm.answers[prompt] = recognize_out_of_order

    result = _first_page(_post(gateway, vlmExtraArgs={"parsingMode": "two-stage"}))

    with fake_vllm.completed_condition:
        assert finished_labels()[-2:] == ["Text", "Title"]

    pruned = result["prunedResult"]
    assert (pruned["width"], pruned["height"]) == PAGE_SIZE
    # Table content is formatted to HTML (tests/test_paddle_format.py covers formatting).
    formatted = {**CONTENT_BY_LABEL, "Table": "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"}
    assert [
        (item["monkey_label"], item["block_bbox"], item["block_content"], item["block_id"], item["block_order"])
        for item in pruned["parsing_res_list"]
    ] == [
        (block["label"], block["bbox"], formatted.get(block["label"], ""), index, index + 1)
        for index, block in enumerate(BLOCKS)
    ]


def test_khối_without_a_prompt_keep_their_bbox_with_empty_content_and_no_request(gateway, fake_vllm):
    result = _first_page(_post(gateway, vlmExtraArgs={"parsingMode": "two-stage"}))

    items = {item["monkey_label"]: item for item in result["prunedResult"]["parsing_res_list"]}
    assert items["Picture"]["block_bbox"] == [372, 1421, 868, 1473]
    assert items["Picture"]["block_content"] == ""
    assert items["Footnote"]["block_bbox"] == [124, 1491, 1116, 1579]
    assert items["Footnote"]["block_content"] == ""
    recognized = {_khối_of(request_image(r["payload"]))["label"] for r in fake_vllm.requests[1:]}
    assert recognized.isdisjoint({"Picture", "Footnote"})


def test_layout_request_upscales_small_pages_to_upstream_min_pixels(gateway, fake_vllm):
    small = Image.new("RGB", (400, 500), "white")
    _first_page(_post(gateway, small, vlmExtraArgs={"parsingMode": "two-stage"}))

    # scale = (1,003,520 / 200,000) ** 0.5 = 2.2399999... in floats; Upstream's
    # int() truncation gives (895, 1119).
    assert request_image(fake_vllm.requests[0]["payload"]).size == (895, 1119)


def test_mode_defaults_to_two_stage_when_env_is_unset(gateway, fake_vllm):
    result = _first_page(_post(gateway))

    assert result["prunedResult"]["model_settings"]["parsing_mode"] == "two-stage"
    assert request_prompt(fake_vllm.requests[0]["payload"]) == LAYOUT_PROMPT


def test_env_sets_the_default_mode_for_requests_without_parsing_mode(fake_vllm, monkeypatch):
    monkeypatch.setenv("MOCR2_PARSING_MODE", "end-to-end")
    with TestClient(create_app()) as client:
        omitted = _first_page(_post(client))
        null_mode = _first_page(_post(client, vlmExtraArgs={"parsingMode": None}))
        overridden = _first_page(_post(client, vlmExtraArgs={"parsingMode": "two-stage"}))

    assert omitted["prunedResult"]["model_settings"]["parsing_mode"] == "end-to-end"
    assert null_mode["prunedResult"]["model_settings"]["parsing_mode"] == "end-to-end"
    assert overridden["prunedResult"]["model_settings"]["parsing_mode"] == "two-stage"
    prompts = [request_prompt(r["payload"]) for r in fake_vllm.requests]
    assert prompts[:3] == [END2END_PROMPT, END2END_PROMPT, LAYOUT_PROMPT]


@pytest.mark.parametrize("mode", ["end2end", "TWO-STAGE", "", 1])
def test_invalid_parsing_mode_is_422_with_clear_message(gateway, fake_vllm, mode):
    response = _post(gateway, vlmExtraArgs={"parsingMode": mode}, logId="client-log")

    assert response.status_code == 422
    body = response.json()
    assert body["logId"] == "client-log"
    assert body["errorCode"] == 422
    assert "parsingMode" in body["errorMsg"]
    assert "two-stage" in body["errorMsg"] and "end-to-end" in body["errorMsg"]
    assert "result" not in body
    assert fake_vllm.requests == []


def test_invalid_default_mode_in_env_fails_at_startup(monkeypatch):
    monkeypatch.setenv("MOCR2_PARSING_MODE", "fast")
    with pytest.raises(ValueError, match="MOCR2_PARSING_MODE"):
        create_app()
