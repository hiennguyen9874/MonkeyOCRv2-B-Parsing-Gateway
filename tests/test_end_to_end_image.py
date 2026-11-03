"""A single Base64 image goes through Chế độ end-to-end and comes back as a PaddleOCR-VL envelope."""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from conftest import END2END_PROMPT, image_base64

# A4 at 150 DPI: larger than the 1,003,520-pixel request limit, so the image
# sent to vLLM is downscaled while bboxes stay in the original page space.
PAGE_SIZE = (1240, 1754)


def _post_page(gateway, fmt="PNG"):
    page = Image.new("RGB", PAGE_SIZE, "white")
    return gateway.post(
        "/layout-parsing",
        json={
            "file": image_base64(page, fmt),
            "fileType": 1,
            "visualize": False,
            "vlmExtraArgs": {"parsingMode": "end-to-end"},
        },
    )


def test_image_returns_one_page_result_in_paddle_envelope(gateway):
    response = _post_page(gateway)

    assert response.status_code == 200
    body = response.json()
    assert body["errorCode"] == 0
    assert body["errorMsg"] == "Success"
    assert isinstance(body["logId"], str) and body["logId"]

    pages = body["result"]["layoutParsingResults"]
    assert len(pages) == 1
    pruned = pages[0]["prunedResult"]
    assert (pruned["width"], pruned["height"]) == PAGE_SIZE
    assert pruned["parsing_res_list"] == [
        {
            "block_label": "Title",
            "block_content": "Quy chế mẫu về bảo mật dữ liệu",
            "block_bbox": [124, 88, 1116, 210],
            "block_id": 0,
            "block_order": 1,
        },
        {
            "block_label": "Text",
            "block_content": "Điều 1. Văn bản này là dữ liệu tổng hợp, chỉ dùng để kiểm thử gateway.",
            "block_bbox": [124, 263, 1116, 702],
            "block_id": 1,
            "block_order": 2,
        },
        {
            "block_label": "Page-footer",
            "block_content": "Trang 1",
            "block_bbox": [99, 1666, 1141, 1719],
            "block_id": 2,
            "block_order": 3,
        },
    ]
    assert pages[0]["markdown"]["text"] == (
        "Quy chế mẫu về bảo mật dữ liệu\n\n"
        "Điều 1. Văn bản này là dữ liệu tổng hợp, chỉ dùng để kiểm thử gateway.\n\n"
        "Trang 1"
    )


def test_jpeg_is_accepted(gateway):
    response = _post_page(gateway, fmt="JPEG")

    assert response.status_code == 200
    pruned = response.json()["result"]["layoutParsingResults"][0]["prunedResult"]
    assert (pruned["width"], pruned["height"]) == PAGE_SIZE


def test_vllm_receives_end2end_prompt_png_data_uri_and_temperature_zero(gateway, fake_vllm):
    _post_page(gateway)

    assert len(fake_vllm.requests) == 1
    request = fake_vllm.requests[0]
    assert request["path"] == "/v1/chat/completions"
    payload = request["payload"]
    assert payload["model"] == "fake-monkey"
    assert payload["temperature"] == 0
    assert "max_tokens" not in payload  # end-to-end has no token limit upstream
    image_part, text_part = payload["messages"][0]["content"]
    assert text_part == {"type": "text", "text": END2END_PROMPT}
    url = image_part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    sent = Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert sent.format == "PNG"
    assert sent.size == (842, 1191)  # resized to the 1,003,520-pixel limit


def test_health_is_healthy_without_calling_vllm(gateway, fake_vllm):
    response = gateway.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["errorCode"] == 0
    assert body["errorMsg"] == "Healthy"
    assert body["logId"]
    assert fake_vllm.requests == []


def test_weknora_ping_get_layout_parsing_is_405(gateway):
    assert gateway.get("/layout-parsing").status_code == 405
