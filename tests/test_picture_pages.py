"""Picture identity across pages and bounded crop retention through the HTTP seam."""

import base64
import json
import weakref
from io import BytesIO

import pytest
from conftest import END2END_PROMPT, LAYOUT_PROMPT
from fake_vllm import FakeVLLM
from fastapi.testclient import TestClient
from PIL import Image

from monkeyocr_gateway.app import create_app


@pytest.mark.parametrize("file_format", ["TIFF", "PDF"])
@pytest.mark.parametrize("mode", ["end-to-end", "two-stage"])
@pytest.mark.parametrize("options", [{}, {"returnMarkdownImages": False}, {"markdownIgnoreLabels": ["image"]}])
def test_picture_pages_keep_distinct_images_without_retaining_raw_crops(monkeypatch, mode, options, file_format):
    # Crop lifetime is not observable in JSON; weak references check the memory
    # bound while the HTTP response verifies identity and suppression.
    crops = []
    live_counts = []
    original_crop = Image.Image.crop

    def tracked_crop(image, box=None):
        crop = original_crop(image, box)
        crops.append(weakref.ref(crop))
        live_counts.append(sum(ref() is not None for ref in crops))
        return crop

    colors = [(200, 0, 0), (0, 200, 0), (0, 0, 200)]
    frames = [Image.new("RGB", (400, 600), color) for color in colors]
    buffer = BytesIO()
    frames[0].save(buffer, file_format, save_all=True, append_images=frames[1:], resolution=200)
    output = json.dumps([{"label": "Picture", "bbox": [0, 0, 1000, 1000], "content": ""}])
    monkeypatch.setenv("MOCR2_PAGE_CONCURRENCY", "1")
    with FakeVLLM({END2END_PROMPT: output, LAYOUT_PROMPT: output}) as fake:
        monkeypatch.setenv("MOCR2_SERVER_URL", fake.url)
        monkeypatch.setenv("MOCR2_SERVED_MODEL_NAME", "fake-monkey")
        monkeypatch.setattr(Image.Image, "crop", tracked_crop)
        with TestClient(create_app()) as client:
            response = client.post("/layout-parsing", json={
                "file": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "fileType": 0 if file_format == "PDF" else 1,
                "vlmExtraArgs": {"parsingMode": mode},
                **options,
            })
    assert response.status_code == 200, response.text
    pages = response.json()["result"]["layoutParsingResults"]
    assert len(pages) == len(colors)
    merged = {}
    for index, (page, color) in enumerate(zip(pages, colors), start=1):
        markdown = page["markdown"]
        if options:
            assert not markdown["images"]
            assert "imgs/" not in markdown["text"]
        else:
            path = f"imgs/page_{index}/img_in_image_box_0_0_400_600.jpg"
            assert list(markdown["images"]) == [path]
            assert f'src="{path}"' in markdown["text"]
            # Mirror WeKnora's first-value-wins document-level merge.
            for key, value in markdown["images"].items():
                merged.setdefault(key, value)
            with Image.open(BytesIO(base64.b64decode(merged[path]))) as crop:
                assert crop.size == (400, 600)
                assert all(abs(a - b) <= 8 for a, b in zip(crop.getpixel((200, 300)), color))
        assert page["prunedResult"]["parsing_res_list"][0]["block_content"] == ""
    assert len(merged) == (0 if options else len(colors))
    if mode == "end-to-end":
        if options:
            assert not crops
        else:
            assert max(live_counts) == 1
    assert all(ref() is None for ref in crops)
