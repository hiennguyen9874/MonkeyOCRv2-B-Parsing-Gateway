"""PaddleOCR-VL option policy (ADR-0002, ADR-0004) and the WeKnora contract test.

WEKNORA_PARAMS copies paddleOCRVLRecognitionParams from WeKnora
internal/infrastructure/docparser/paddleocr_vl_converter.go (02e30e9f5), with
seal and chart recognition on, as tenants can configure them.
"""

from __future__ import annotations

import base64
import re
from urllib.parse import unquote

import pytest
from pydantic import BaseModel, ConfigDict, Field

from conftest import END2END_PROMPT, LAYOUT_PROMPT, TEXT_PROMPT, image_base64
from fake_vllm import request_image, request_prompt
from test_paddle_format import _painted_page, fake_vllm  # noqa: F401 - fixture

WEKNORA_PARAMS = {
    "markdownIgnoreLabels": [
        "header", "header_image", "footer", "footer_image",
        "number", "footnote", "aside_text",
    ],
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useLayoutDetection": True,
    "useChartRecognition": True,
    "useSealRecognition": True,
    "useOcrForImageBlock": False,
    "mergeTables": True,
    "relevelTitles": True,
    "restructurePages": True,
    "layoutShapeMode": "auto",
    "promptLabel": "ocr",
    "layoutNms": True,
    "repetitionPenalty": 1,
    "temperature": 0,
    "topP": 1,
    "minPixels": 147384,
    "maxPixels": 2822400,
}
MODES = ["two-stage", "end-to-end"]


# Mirrors of WeKnora's Go structs. Strict mode rejects the type mismatches that
# make Go's json.Unmarshal fail (for example a string where a number is expected).
class _Go(BaseModel):
    model_config = ConfigDict(strict=True)


class PaddleOCRVLLayoutBlock(_Go):
    block_content: str | None = None
    block_bbox: list[float] | None = None


class PaddleOCRVLLayout(_Go):
    width: float | None = None
    height: float | None = None
    parsing_res_list: list[PaddleOCRVLLayoutBlock] | None = None


class PaddleOCRVLMarkdown(_Go):
    text: str | None = None
    images: dict[str, str] | None = None


class PaddleOCRVLPage(_Go):
    markdown: PaddleOCRVLMarkdown | None = None
    prunedResult: dict | None = None  # json.RawMessage, decoded as paddleOCRVLLayout


class PaddleOCRVLResult(_Go):
    layoutParsingResults: list[PaddleOCRVLPage] | None = None


class PaddleOCRVLResponse(_Go):
    errorCode: int = 0
    errorMsg: str | None = None
    result: PaddleOCRVLResult | None = Field(default=None)


# WeKnora extractImageRefsFromContent (mineru_converter.go) with
# searchutil.HTMLImageSrcRegex: Markdown images and HTML <img src>.
_MARKDOWN_IMAGE = re.compile(r"!\[(.*?)\]\(([^()\n]*(?:\([^)]*\)[^()\n]*)*)\)")
_HTML_IMAGE = re.compile(r"""(?i)<img\b([^>]*?)\ssrc\s*=\s*['"]([^'"]+)['"]([^>]*)>""")


def _image_refs(markdown: str) -> list[str]:
    return [m[1].strip() for m in _MARKDOWN_IMAGE.findall(markdown)] + [
        m[1] for m in _HTML_IMAGE.findall(markdown)
    ]


def _normalize(path: str) -> str:
    # WeKnora normalizeMinerUImagePath, used to match references to image keys.
    path = unquote(path.strip()) or path.strip()
    return path.removeprefix("./").removeprefix("/").removeprefix("images/")


def _post(gateway, mode="two-stage", **fields):
    return gateway.post("/layout-parsing", json={
        "file": image_base64(_painted_page()),
        "fileType": 1,
        "vlmExtraArgs": {"parsingMode": mode},
        **fields,
    })


def _weknora_request(mode=None) -> dict:
    # Exactly what callLayoutParsing sends; parsingMode is added only to pick a mode.
    body = {**WEKNORA_PARAMS, "file": image_base64(_painted_page()), "fileType": 1, "visualize": False}
    if mode is not None:
        body["vlmExtraArgs"] = {"parsingMode": mode}
    return body


@pytest.mark.parametrize("mode", [None, *MODES])
def test_weknora_request_succeeds_and_decodes_with_weknora_structs(gateway, mode):
    response = gateway.post("/layout-parsing", json=_weknora_request(mode))

    assert response.status_code == 200, response.text
    decoded = PaddleOCRVLResponse.model_validate(response.json())
    assert decoded.errorCode == 0
    pages = decoded.result.layoutParsingResults
    assert len(pages) == 1
    layout = PaddleOCRVLLayout.model_validate(pages[0].prunedResult)
    assert layout.width > 0 and layout.height > 0 and layout.parsing_res_list
    for block in layout.parsing_res_list:
        x1, y1, x2, y2 = block.block_bbox
        assert 0 <= x1 < x2 <= layout.width and 0 <= y1 < y2 <= layout.height

    refs = _image_refs(pages[0].markdown.text)
    assert refs, "the synthetic page has a Picture"
    assert {_normalize(ref) for ref in refs} <= {_normalize(key) for key in pages[0].markdown.images}
    for data in pages[0].markdown.images.values():
        base64.b64decode(data, validate=True)


@pytest.mark.parametrize("mode", MODES)
def test_model_settings_report_actual_behaviour_not_the_requested_flags(gateway, mode):
    response = gateway.post("/layout-parsing", json=_weknora_request(mode))

    settings = response.json()["result"]["layoutParsingResults"][0]["prunedResult"]["model_settings"]
    assert settings == {
        "use_doc_preprocessor": False,
        "use_layout_detection": True,
        "use_chart_recognition": False,
        "use_seal_recognition": False,
        "use_ocr_for_image_block": False,
        "format_block_content": False,
        "merge_layout_blocks": False,
        "restructure_pages": False,
        "markdown_ignore_labels": WEKNORA_PARAMS["markdownIgnoreLabels"],
        "parsing_mode": mode,
    }


@pytest.mark.parametrize("flag", ["useDocOrientationClassify", "useDocUnwarping"])
def test_preprocessing_true_is_rejected_without_calling_vllm(gateway, fake_vllm, flag):
    response = _post(gateway, logId="client-log-1", **{flag: True})

    assert response.status_code == 422
    body = response.json()
    assert body["errorCode"] == 422
    assert body["logId"] == "client-log-1"
    assert flag in body["errorMsg"] and "not supported" in body["errorMsg"]
    assert body.get("result") is None
    assert fake_vllm.requests == []


@pytest.mark.parametrize("value", [False, None])
def test_preprocessing_false_or_null_is_accepted(gateway, value):
    response = _post(gateway, useDocOrientationClassify=value, useDocUnwarping=value)

    assert response.status_code == 200, response.text


@pytest.mark.parametrize("mode", MODES)
def test_request_min_max_pixels_do_not_change_the_image_sent_to_vllm(gateway, fake_vllm, mode):
    _post(gateway, mode)
    baseline = [(request_prompt(r["payload"]), request_image(r["payload"]).size) for r in fake_vllm.requests]
    fake_vllm.requests.clear()

    _post(gateway, mode, minPixels=2_000_000, maxPixels=10_000)
    with_pixels = [(request_prompt(r["payload"]), request_image(r["payload"]).size) for r in fake_vllm.requests]

    assert sorted(with_pixels) == sorted(baseline)


@pytest.mark.parametrize("mode", MODES)
def test_request_generation_options_reach_vllm(gateway, fake_vllm, mode):
    response = _post(gateway, mode, temperature=0.3, topP=0.8, repetitionPenalty=1.05, maxNewTokens=777)

    assert response.status_code == 200, response.text
    assert fake_vllm.requests
    for request in fake_vllm.requests:
        payload = request["payload"]
        assert payload["temperature"] == 0.3
        assert payload["top_p"] == 0.8
        assert payload["repetition_penalty"] == 1.05
        assert payload["max_tokens"] == 777


@pytest.mark.parametrize("mode", MODES)
def test_null_generation_options_use_upstream_defaults(gateway, fake_vllm, mode):
    response = _post(gateway, mode, temperature=None, topP=None, repetitionPenalty=None, maxNewTokens=None)

    assert response.status_code == 200, response.text
    expected_max_tokens = {LAYOUT_PROMPT: 4096, END2END_PROMPT: None}
    for request in fake_vllm.requests:
        payload = request["payload"]
        assert payload["temperature"] == 0
        assert "top_p" not in payload
        assert "repetition_penalty" not in payload
        assert payload.get("max_tokens") == expected_max_tokens.get(request_prompt(payload), 5000)
    prompts = {request_prompt(r["payload"]) for r in fake_vllm.requests}
    assert (TEXT_PROMPT in prompts) == (mode == "two-stage")


def test_client_log_id_is_returned(gateway):
    response = _post(gateway, logId="weknora-123")

    assert response.status_code == 200
    assert response.json()["logId"] == "weknora-123"


def test_log_id_is_generated_when_missing(gateway):
    first = _post(gateway).json()["logId"]
    second = _post(gateway).json()["logId"]

    assert first and second and first != second


@pytest.mark.parametrize("fields", [
    {"vlmExtraArgs": {"parsingMode": "bogus"}},  # rejected by the handler
    {"temperature": "hot"},  # rejected by request validation
    {"fileType": 7},  # rejected by the document loader
])
def test_errors_keep_the_client_log_id(gateway, fields):
    body = {"file": image_base64(_painted_page()), "fileType": 1, "logId": "err-1", **fields}

    response = gateway.post("/layout-parsing", json=body)

    assert response.status_code == 422
    assert response.json()["logId"] == "err-1"
    assert response.json()["errorCode"] == 422


@pytest.mark.parametrize("field", ["temperature", "topP", "repetitionPenalty"])
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_generation_options_are_rejected(gateway, fake_vllm, field, value):
    response = _post(gateway, logId="invalid-generation", **{field: value})

    assert response.status_code == 422
    body = response.json()
    assert body["errorCode"] == 422
    assert body["logId"] == "invalid-generation"
    assert field in body["errorMsg"]
    assert body["result"] is None
    assert fake_vllm.requests == []


def test_validation_error_without_log_id_gets_a_generated_one_and_the_envelope(gateway):
    response = gateway.post("/layout-parsing", json={"fileType": 1})

    assert response.status_code == 422
    body = response.json()
    assert body["logId"]
    assert body["errorCode"] == 422
    assert "file" in body["errorMsg"]
    assert body.get("result") is None


@pytest.mark.parametrize("mode", MODES)
def test_unsupported_image_fields_are_null(gateway, mode):
    response = gateway.post("/layout-parsing", json={**_weknora_request(mode), "visualize": True})

    page = response.json()["result"]["layoutParsingResults"][0]
    assert page["outputImages"] is None and page["inputImage"] is None
    assert "layout_det_res" not in page["prunedResult"]
    assert "exports" not in page
