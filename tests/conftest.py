from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from fake_vllm import FakeVLLM
from monkeyocr_gateway.app import create_app

FIXTURES = Path(__file__).parent / "fixtures"
# Prompts copied from Upstream core_runner.ALL_PROMPT at 106914e.
END2END_PROMPT = (
    "List the document elements in reading order, including their categories, "
    "coordinates, and the content of each element."
)
LAYOUT_PROMPT = "Please output the categories and coordinates of the document elements in reading order."
TEXT_PROMPT = "Please output the text content from the image."
FORMULA_PROMPT = "Please write out the expression of the formula in the image using LaTeX format."
TABLE_PROMPT = "Please extract the table from the image and represent it in OTSL format."


def recorded(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def image_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.fixture
def fake_vllm(monkeypatch):
    with FakeVLLM({END2END_PROMPT: recorded("end2end_synthetic_page.txt")}) as fake:
        monkeypatch.setenv("MOCR2_SERVER_URL", fake.url)
        monkeypatch.setenv("MOCR2_SERVED_MODEL_NAME", "fake-monkey")
        yield fake


@pytest.fixture
def gateway(fake_vllm):
    with TestClient(create_app()) as client:
        yield client
