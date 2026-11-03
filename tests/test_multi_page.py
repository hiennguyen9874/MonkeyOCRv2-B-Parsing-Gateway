"""Multi-page documents: PDF, multi-frame TIFF, file type detection, dataInfo and input limits."""

from __future__ import annotations

import base64
import json
import time
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import END2END_PROMPT, image_base64
from fake_vllm import FakeVLLM, request_image
from monkeyocr_gateway.app import create_app

# Each page is one solid colour so the fake model can tell which page it received.
COLORS = [
    (200, 0, 0), (0, 200, 0), (0, 0, 200), (200, 200, 0), (0, 200, 200),
    (200, 0, 200), (90, 90, 90), (150, 60, 0), (0, 60, 150), (60, 150, 0),
]
SIZES = [(400, 600), (600, 400), (300, 500)]


def _pages(count: int, sizes=SIZES) -> list[Image.Image]:
    return [Image.new("RGB", sizes[i % len(sizes)], COLORS[i]) for i in range(count)]


def _page_number(image: Image.Image) -> int:
    """1-based page number from the page colour (PDF rendering may shift it slightly)."""
    pixel = image.convert("RGB").getpixel((image.width // 2, image.height // 2))
    distances = [sum((a - b) ** 2 for a, b in zip(pixel, color)) for color in COLORS]
    return distances.index(min(distances)) + 1


def _end2end_answer(image: Image.Image) -> str:
    return json.dumps([
        {"bbox": [0, 0, 1000, 1000], "label": "Text", "content": f"Trang {_page_number(image)}"},
    ])


def _pdf_base64(pages: list[Image.Image]) -> str:
    buffer = BytesIO()
    # At 200 DPI one image pixel is one pixel of the page rendered at Upstream's 200 DPI.
    pages[0].save(buffer, "PDF", save_all=True, append_images=pages[1:], resolution=200)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _tiff_base64(frames: list[Image.Image]) -> str:
    buffer = BytesIO()
    frames[0].save(buffer, "TIFF", save_all=True, append_images=frames[1:])
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.fixture
def fake_vllm(monkeypatch):
    with FakeVLLM({END2END_PROMPT: _end2end_answer}) as fake:
        monkeypatch.setenv("MOCR2_SERVER_URL", fake.url)
        monkeypatch.setenv("MOCR2_SERVED_MODEL_NAME", "fake-monkey")
        for name in ("MOCR2_PAGE_CONCURRENCY", "MOCR2_MAX_PAGES", "MOCR2_MAX_BODY_BYTES"):
            monkeypatch.delenv(name, raising=False)
        yield fake


def _client():
    return TestClient(create_app())


def _post(client, file: str, file_type=None, **extra):
    return client.post(
        "/layout-parsing",
        json={
            "file": file,
            "fileType": file_type,
            "vlmExtraArgs": {"parsingMode": "end-to-end"},
            **extra,
        },
    )


def _ok(response) -> dict:
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["errorCode"] == 0
    return body["result"]


def _contents(result) -> list[str]:
    return [
        page["prunedResult"]["parsing_res_list"][0]["block_content"]
        for page in result["layoutParsingResults"]
    ]


def _sizes(result) -> list[tuple[int, int]]:
    return [
        (page["prunedResult"]["width"], page["prunedResult"]["height"])
        for page in result["layoutParsingResults"]
    ]


def _error(response, status: int, *words: str) -> dict:
    assert response.status_code == status, response.text
    body = response.json()
    assert body["errorCode"] == status
    assert body["result"] is None
    assert body["logId"]
    for word in words:
        assert word in body["errorMsg"]
    return body


@pytest.mark.parametrize("file_type", [0, None])
def test_pdf_pages_come_back_in_page_order_when_they_finish_out_of_order(fake_vllm, file_type):
    def finished_pages():
        return [_page_number(request_image(p)) for p in fake_vllm.completed]

    def answer_out_of_order(image):
        prerequisites = {1: {2, 3}, 2: {3}}.get(_page_number(image), set())
        with fake_vllm.completed_condition:
            assert fake_vllm.completed_condition.wait_for(
                lambda: prerequisites.issubset(finished_pages()), timeout=5
            ), f"Page stalled waiting for {prerequisites}"
        return _end2end_answer(image)

    fake_vllm.answers[END2END_PROMPT] = answer_out_of_order
    with _client() as client:
        result = _ok(_post(client, _pdf_base64(_pages(3)), file_type))

    with fake_vllm.completed_condition:
        assert finished_pages() == [3, 2, 1]
    assert _contents(result) == ["Trang 1", "Trang 2", "Trang 3"]
    assert _sizes(result) == SIZES
    assert [page["prunedResult"]["page_count"] for page in result["layoutParsingResults"]] == [3, 3, 3]
    assert result["dataInfo"] == {
        "type": "pdf",
        "numPages": 3,
        "pages": [{"width": w, "height": h} for w, h in SIZES],
    }


@pytest.mark.parametrize("file_type", [1, None])
def test_multi_frame_tiff_gives_one_page_result_per_frame(fake_vllm, file_type):
    with _client() as client:
        result = _ok(_post(client, _tiff_base64(_pages(3)), file_type))

    assert _contents(result) == ["Trang 1", "Trang 2", "Trang 3"]
    assert _sizes(result) == SIZES
    # As in PaddleOCR-VL, page_count is only set for PDF input.
    assert [page["prunedResult"]["page_count"] for page in result["layoutParsingResults"]] == [None] * 3
    assert result["dataInfo"] == {"type": "image", "width": 400, "height": 600}


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "BMP", "WEBP"])
def test_image_without_file_type_is_detected(fake_vllm, fmt):
    with _client() as client:
        result = _ok(_post(client, image_base64(_pages(1)[0], fmt)))

    assert _contents(result) == ["Trang 1"]
    assert result["layoutParsingResults"][0]["prunedResult"]["page_count"] is None
    assert result["dataInfo"] == {"type": "image", "width": 400, "height": 600}


def test_jpeg_exif_orientation_is_applied(fake_vllm):
    image = Image.new("RGB", (400, 600), COLORS[0])
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° clockwise to display
    buffer = BytesIO()
    image.save(buffer, "JPEG", exif=exif)
    with _client() as client:
        result = _ok(_post(client, base64.b64encode(buffer.getvalue()).decode("ascii"), 1))

    assert _sizes(result) == [(600, 400)]
    assert result["dataInfo"] == {"type": "image", "width": 600, "height": 400}


@pytest.mark.parametrize(
    ("file", "file_type", "words"),
    [
        (base64.b64encode(b"hello, not a document").decode(), None, ("file type",)),
        (base64.b64encode(b"hello, not a document").decode(), 1, ("image",)),
        (base64.b64encode(b"%PDF-1.4 broken").decode(), 0, ("PDF",)),
        (image_base64(Image.new("RGB", (10, 10))), 0, ("PDF",)),
        (image_base64(Image.new("RGB", (10, 10))), 2, ("fileType",)),
        ("not base64!", 1, ("Base64",)),
        ("", None, ("Base64",)),
    ],
)
def test_bad_input_is_422_without_calling_vllm(fake_vllm, file, file_type, words):
    with _client() as client:
        response = _post(client, file, file_type, logId="client-log")

    body = _error(response, 422, *words)
    assert body["logId"] == "client-log"
    assert fake_vllm.requests == []


@pytest.mark.parametrize(("env", "pages", "expected"), [(None, 10, 8), ("2", 6, 2)])
def test_pages_in_parallel_do_not_exceed_the_configured_limit(fake_vllm, monkeypatch, env, pages, expected):
    if env is not None:
        monkeypatch.setenv("MOCR2_PAGE_CONCURRENCY", env)

    def slow_answer(image):
        time.sleep(0.2)
        return _end2end_answer(image)

    fake_vllm.answers[END2END_PROMPT] = slow_answer
    with _client() as client:
        result = _ok(_post(client, _pdf_base64(_pages(pages))))

    assert len(result["layoutParsingResults"]) == pages
    assert fake_vllm.max_in_flight == expected


@pytest.mark.parametrize("encode", [_pdf_base64, _tiff_base64])
def test_page_limit(fake_vllm, monkeypatch, encode):
    monkeypatch.setenv("MOCR2_MAX_PAGES", "2")
    with _client() as client:
        too_many = _post(client, encode(_pages(3)))
        _error(too_many, 413, "3 pages", "2")
        assert fake_vllm.requests == []
        assert len(_ok(_post(client, encode(_pages(2))))["layoutParsingResults"]) == 2


def test_body_size_limit(fake_vllm, monkeypatch):
    monkeypatch.setenv("MOCR2_MAX_BODY_BYTES", "2000")
    small = image_base64(Image.new("RGB", (10, 10)))
    large = image_base64(Image.effect_noise((100, 100), 64))
    with _client() as client:
        _error(_post(client, large, 1), 413, "2000 bytes")
        # Without Content-Length (chunked upload) the limit still applies.
        payload = json.dumps({"file": large, "fileType": 1}).encode()
        chunked = client.post(
            "/layout-parsing",
            content=(payload[i:i + 500] for i in range(0, len(payload), 500)),
            headers={"Content-Type": "application/json"},
        )
        _error(chunked, 413, "2000 bytes")
        assert fake_vllm.requests == []
        assert _ok(_post(client, small, 1))
