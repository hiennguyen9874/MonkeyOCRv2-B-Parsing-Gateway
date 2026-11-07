"""Chế độ định tuyến theo trang (parsingMode "routed", ADR-0007) through the HTTP API.

Synthetic PDFs only. The fake inference server answers layout with boxes computed
from the PDF geometry, so each test checks where OCR Khối land on the page.
"""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import pymupdf
import pytest
from conftest import FORMULA_PROMPT, image_base64, LAYOUT_PROMPT, TABLE_PROMPT, TEXT_PROMPT
from fake_vllm import FakeVLLM, request_image, request_prompt
from fastapi.testclient import TestClient
from PIL import Image
from router_pdfs import DIAGRAM, diagram, table_pdf, text_image_png

from monkeyocr_gateway.app import create_app
from monkeyocr_gateway.document_loader import PdfDocument
from monkeyocr_gateway.routed import NativeDocument
from monkeyocr_gateway.router.gate import assess


@pytest.fixture
def fake_vllm(monkeypatch):
    answers = {LAYOUT_PROMPT: "[]", TEXT_PROMPT: "", TABLE_PROMPT: "", FORMULA_PROMPT: ""}
    with FakeVLLM(answers) as fake:
        monkeypatch.setenv("MOCR2_SERVER_URL", fake.url)
        monkeypatch.setenv("MOCR2_SERVED_MODEL_NAME", "fake-monkey")
        monkeypatch.delenv("MOCR2_PARSING_MODE", raising=False)
        yield fake


def _post(client, data: bytes, file_type=0, **extra):
    body = {"file": base64.b64encode(data).decode(), "fileType": file_type,
            "vlmExtraArgs": {"parsingMode": "routed"}, **extra}
    response = client.post("/layout-parsing", json=body)
    assert response.status_code == 200, response.text
    return response.json()["result"]["layoutParsingResults"]


def _prompts(fake) -> list[str]:
    return [request_prompt(r["payload"]) for r in fake.requests]


def _codes(page) -> list[str]:
    return [r["code"] for r in page["prunedResult"]["routing"]["review_reasons"]]


def _to_pixels(page: pymupdf.Page, bbox, size) -> list[float]:
    """Unrotated PDF points -> rendered page pixels."""
    r = pymupdf.Rect(bbox) * page.rotation_matrix
    sx, sy = size[0] / page.rect.width, size[1] / page.rect.height
    return [r.x0 * sx, r.y0 * sy, r.x1 * sx, r.y1 * sy]


def _layout_box(page: pymupdf.Page, region, bbox, padding_pixels=0) -> list[int]:
    """bbox (unrotated points) in the 0-1000 layout space of the region's OCR crop."""
    pad = padding_pixels * 72 / 200
    visual = pymupdf.Rect(region) * page.rotation_matrix + (-pad, -pad, pad, pad)
    r = pymupdf.Rect(bbox) * page.rotation_matrix
    return [round((r.x0 - visual.x0) / visual.width * 1000), round((r.y0 - visual.y0) / visual.height * 1000),
            round((r.x1 - visual.x0) / visual.width * 1000), round((r.y1 - visual.y0) / visual.height * 1000)]


def _blocks(page):
    return page["prunedResult"]["parsing_res_list"]


def test_native_page_uses_text_layer_without_inference(gateway, fake_vllm):
    with pymupdf.open() as doc:
        p = doc.new_page()
        p.insert_text((50, 60), "Native legal heading with enough valid characters")
        p.insert_text((50, 100), "Second native paragraph line")
        data = doc.tobytes()
    (page,) = _post(gateway, data)

    assert fake_vllm.requests == []
    pruned = page["prunedResult"]
    routing = pruned["routing"]
    assert routing["decision"] == "native"
    assert routing["effective_mode"] == "native-only"
    assert routing["reasons"] == ["native_quality_gate_passed"]
    assert routing["page_type"] == "native"
    assert "invisible_ratio" in routing["signals"] and "max_bad_glyph_ratio" in routing["thresholds"]
    assert routing["requests"] == []
    assert pruned["model_settings"]["parsing_mode"] == "routed"
    assert pruned["model_settings"]["page_router"]["ocr_parsing_mode"] == "two-stage"
    texts = [b["block_content"] for b in _blocks(page)]
    assert texts == ["Native legal heading with enough valid characters", "Second native paragraph line"]
    assert all(b["block_source"] == "native" and b["block_label"] == "text" for b in _blocks(page))
    assert page["markdown"]["text"] == "\n\n".join(texts)


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_native_khoi_bbox_matches_the_rendered_page(gateway, fake_vllm, angle):
    """PyMuPDF text geometry agrees with the pdfium render, with /Rotate and CropBox."""
    with pymupdf.open() as doc:
        p = doc.new_page(width=300, height=400)
        p.insert_text((40, 80), "Native heading with enough valid characters", fontsize=11)
        if angle == 270:
            p.set_cropbox(pymupdf.Rect(10, 10, 290, 390))
        p.set_rotation(angle)
        data = doc.tobytes()
    (page,) = _post(gateway, data)
    (block,) = _blocks(page)
    rendered = PdfDocument(data)
    ink = np.argwhere(np.asarray(rendered.page(0).convert("L")) < 128)
    rendered.close()
    (y0, x0), (y1, x1) = ink.min(0), ink.max(0)
    bx0, by0, bx1, by1 = block["block_bbox"]
    assert bx0 - 3 <= x0 and by0 - 3 <= y0 and x1 <= bx1 + 3 and y1 <= by1 + 3
    assert (bx1 - bx0) * (by1 - by0) < 3 * (x1 - x0) * (y1 - y0)


def _ruled_table(page, top):
    for y in (top, top + 40, top + 80):
        page.draw_line((50, y), (250, y))
    for x in (50, 150, 250):
        page.draw_line((x, top), (x, top + 80))
    for x, y, text in ((55, 25, "Name"), (155, 25, "Value"), (55, 65, "alpha"), (155, 65, "beta")):
        page.insert_text((x, top + y), text, fontsize=10)


def test_table_detection_on_a_rotated_page_does_not_move_its_khoi(gateway, fake_vllm):
    """PyMuPDF find_tables() rewrites rotated pages; it also runs on the previous page
    when a page starts with a table. Neither may shift Khối of the rotated page."""
    with pymupdf.open() as doc:
        p = doc.new_page(width=300, height=400)
        p.insert_text((40, 60), "Rotated page heading with enough characters", fontsize=10)
        _ruled_table(p, 250)
        p.set_cropbox(pymupdf.Rect(10, 10, 290, 390))
        p.set_rotation(90)
        alone = doc.tobytes()
        following = doc.new_page(width=300, height=400)
        _ruled_table(following, 30)
        following.insert_text((50, 200), "A closing paragraph below the table", fontsize=10)
        both = doc.tobytes()
    # Planning the next page first runs find_tables() on the rotated page.
    native = NativeDocument(both)
    native.plan(1)
    planned_after = native.plan(0)
    planned_alone = NativeDocument(alone).plan(0)
    for key in ("rect", "bounds", "rotation_matrix", "word_boxes"):
        assert planned_after[key] == planned_alone[key]
    assert [b["bbox"] for b in planned_after["blocks"]] == [b["bbox"] for b in planned_alone["blocks"]]
    first = _post(gateway, alone)[0]
    # Tables are not detected on rotated pages; their text stays as text Khối.
    assert first["prunedResult"]["routing"]["native_tables_skipped"] == "rotated_page"
    assert [b["block_label"] for b in _blocks(first)] == ["text"] * 3
    assert "alpha" in first["markdown"]["text"]
    heading = _blocks(first)[0]
    assert heading["block_content"].startswith("Rotated page heading")
    rendered = PdfDocument(alone)
    gray = np.asarray(rendered.page(0).convert("L"))
    rendered.close()
    x0, y0, x1, y1 = heading["block_bbox"]
    assert (gray[y0:y1, x0:x1] < 128).any()


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_supplement_masks_native_text_maps_rotated_boxes_and_drops_duplicates(gateway, fake_vllm, angle):
    with pymupdf.open() as doc:
        p = doc.new_page(width=300, height=320)
        p.insert_text((20, 40), "Native heading with enough valid characters")
        p.insert_image(pymupdf.Rect(30, 100, 270, 230), stream=text_image_png())
        p.insert_text((40, 150), "Native repeats")
        if angle == 270:
            p.set_cropbox(pymupdf.Rect(10, 10, 290, 310))
        p.set_rotation(angle)
        data = doc.tobytes()
        p = doc.load_page(0)
        (region,) = assess(p)["supplement_regions"]
        repeats = next(w[:4] for w in p.get_text("words") if w[4] == "repeats")
        duplicate = next(b[:4] for b in p.get_text("blocks") if "Native repeats" in b[4])
        target = [50, 170, 180, 190]
        layout = repr([{"label": "Text", "bbox": _layout_box(p, region, target)},
                       {"label": "Text", "bbox": _layout_box(p, region, duplicate)}])
        repeats_box = _layout_box(p, region, repeats)
        masked = []

        def answer_layout(image):
            # Runs on the fake server thread: no PyMuPDF calls here.
            rgb = np.asarray(image.convert("L"))
            h, w = rgb.shape
            x0, y0, x1, y1 = repeats_box
            inner = rgb[int(y0 * h / 1000) + 3:int(y1 * h / 1000) - 3, int(x0 * w / 1000) + 3:int(x1 * w / 1000) - 3]
            masked.append((inner.size > 0 and bool((inner > 250).all()), bool((rgb < 128).any())))
            return layout

        fake_vllm.answers[LAYOUT_PROMPT] = answer_layout
        fake_vllm.answers[TEXT_PROMPT] = "Supplement only"
        (page,) = _post(gateway, data)
        expected = _to_pixels(p, target, (page["prunedResult"]["width"], page["prunedResult"]["height"]))

    assert masked == [(True, True)]  # native word white, image text still sent
    routing = page["prunedResult"]["routing"]
    assert routing["decision"] == "native"
    assert routing["effective_mode"] == "native+ocr"
    assert "uncovered_image_ink" in routing["reasons"]
    (request,) = routing["requests"]
    assert request["masked_native_words"] and request["dropped_spatial_duplicates"] == 1
    supplement = [b for b in _blocks(page) if b["block_source"] == "ocr"]
    assert [b["block_content"] for b in supplement] == ["Supplement only"]
    assert supplement[0]["block_review"]
    for actual, wanted in zip(supplement[0]["block_bbox"], expected):
        assert abs(actual - wanted) <= 3
    assert sum("Native repeats" in b["block_content"] for b in _blocks(page)) == 1
    assert {"supplemental_ocr", "spatial_duplicates_dropped"} <= set(_codes(page))


def test_small_image_stamp_faint_and_blank_pages_are_not_silently_lost(gateway, fake_vllm):
    stamp_doc = pymupdf.open()
    sp = stamp_doc.new_page(width=180, height=80)
    sp.draw_circle((70, 40), 25, color=(1, 0, 0), width=2)
    stamp = sp.get_pixmap(dpi=150, alpha=False).tobytes("png")
    small = text_image_png(180, 80, ["Code TEST-042", "Amount 250.000", "Date 06/11/2026"], fontsize=9,
                           origin=(8, 18), step=20)
    faint = text_image_png(595, 842, ["FAINT TEST-001"], fontsize=9, color=(0.85,) * 3, origin=(8, 18))
    with pymupdf.open() as doc:
        for image in (small, stamp):
            p = doc.new_page(width=595, height=842)
            p.insert_text((55, 75), "Native heading with enough legal characters")
            p.insert_text((55, 700), "Native footer must follow image text")
            p.insert_image(pymupdf.Rect(60, 240, 240, 320), stream=image)
        doc.new_page(width=595, height=842).insert_image(pymupdf.Rect(0, 0, 595, 842), stream=faint)
        doc.new_page(width=595, height=842)
        data = doc.tobytes()
        region = assess(doc[0])["supplement_regions"][0]
        layout_small = repr([{"label": "Text", "bbox": _layout_box(doc[0], region, [68, 248, 180, 290], 33)}])
    first_rows = []

    def answer_layout(image):
        rgb = np.asarray(image.convert("L"))
        if rgb.shape[0] < 1000:  # the small crop, upscaled for layout
            first_rows.append(bool((rgb[0] >= 250).all()))
            return layout_small
        return "[{'label': 'Text', 'bbox': [10, 10, 990, 990]}]"

    fake_vllm.answers[LAYOUT_PROMPT] = answer_layout
    fake_vllm.answers[TEXT_PROMPT] = lambda image: "Code TEST-042" if image.height < 400 else "FAINT TEST-001"
    small_page, stamp_page, faint_page, blank_page = _post(gateway, data)

    # Small textual image: OCR with white padding, placed before the native footer.
    routing = small_page["prunedResult"]["routing"]
    assert routing["effective_mode"] == "native+ocr"
    assert routing["requests"][0]["padding_pixels"] > 0
    assert first_rows == [True]
    assert [b["block_content"] for b in _blocks(small_page)] == [
        "Native heading with enough legal characters", "Code TEST-042", "Native footer must follow image text"]
    assert abs(_blocks(small_page)[1]["block_bbox"][0] - 68 * 200 / 72) <= 3

    # Coloured mark: no OCR, retained as a reviewed Picture with its crop in the Markdown.
    routing = stamp_page["prunedResult"]["routing"]
    assert routing["effective_mode"] == "native-only" and routing["requests"] == []
    assert "unrecognized_small_image" in _codes(stamp_page)
    picture = next(b for b in _blocks(stamp_page) if b["block_label"] == "image")
    assert picture["block_source"] == "routing" and picture["block_review"]
    assert stamp_page["markdown"]["images"]
    assert stamp_page["markdown"]["text"].index("<img") < stamp_page["markdown"]["text"].index("footer")

    # Faint scan: measurable ink reaches full-page OCR and is reviewed.
    routing = faint_page["prunedResult"]["routing"]
    assert routing["decision"] == "ocr" and not routing["near_blank"]
    assert routing["effective_mode"] == "full-ocr" and routing["page_type"] == "scan"
    assert "low_contrast_ocr_required" in _codes(faint_page)
    assert [b["block_content"] for b in _blocks(faint_page)] == ["FAINT TEST-001"]

    # Ink-free page: no OCR, one empty reviewed Picture.
    routing = blank_page["prunedResult"]["routing"]
    assert routing["effective_mode"] == "near-blank-review" and routing["requests"] == []
    assert "near_blank_retained_without_recognition" in routing["reasons"]
    assert routing["review"]
    assert [b["block_label"] for b in _blocks(blank_page)] == ["image"]
    assert _prompts(fake_vllm).count(LAYOUT_PROMPT) == 2  # small crop + faint page


def test_hidden_text_layer_goes_to_full_page_ocr(gateway, fake_vllm):
    with pymupdf.open() as doc:
        p = doc.new_page(width=300, height=300)
        p.insert_text((20, 40), "Hidden broken text layer with enough characters", render_mode=3)
        p.insert_text((20, 100), "Real visible content")
        data = doc.tobytes()
    fake_vllm.answers[LAYOUT_PROMPT] = "[{'label': 'Text', 'bbox': [50, 50, 950, 400]}]"
    fake_vllm.answers[TEXT_PROMPT] = "OCR of the visible page"
    (page,) = _post(gateway, data)

    routing = page["prunedResult"]["routing"]
    assert routing["decision"] == "ocr" and "invisible_text" in routing["reasons"]
    assert routing["page_type"] == "native"  # a text layer exists, but is not trusted
    assert [r["region"] for r in routing["requests"]] == [None]
    layout = next(r["payload"] for r in fake_vllm.requests if request_prompt(r["payload"]) == LAYOUT_PROMPT)
    assert request_image(layout).size[0] > 0
    assert [(b["block_content"], b["block_source"]) for b in _blocks(page)] == [("OCR of the visible page", "ocr")]


def test_native_table_is_html_table_khoi(gateway, fake_vllm):
    with table_pdf() as doc:
        data = doc.tobytes()
    (page,) = _post(gateway, data)

    assert fake_vllm.requests == []
    labels = [b["block_label"] for b in _blocks(page)]
    assert labels == ["text", "table", "text"]
    table = _blocks(page)[1]
    assert table["block_content"].startswith("<table><tr><th")
    assert '<th rowspan="1" colspan="2">Wide header</th>' in table["block_content"]
    assert table["block_content"] in page["markdown"]["text"]
    assert "table_requires_review" in _codes(page)


@pytest.mark.parametrize("truncated", [False, True])
def test_textual_picture_in_supplement_is_recovered_with_text_prompt(gateway, fake_vllm, truncated):
    png = BytesIO()
    Image.fromarray(diagram()).save(png, format="PNG")
    with pymupdf.open() as doc:
        p = doc.new_page(width=400, height=500)
        p.insert_text((20, 40), "Native legal heading remains unchanged")
        p.insert_text((20, 460), "Native footer remains last")
        p.insert_image((50, 150, 350, 310), stream=png.getvalue())
        data = doc.tobytes()
    fake_vllm.answers[LAYOUT_PROMPT] = "[{'label': 'Picture', 'bbox': [0, 0, 1000, 1000]}]"
    def recognize(_image):
        fake_vllm.finish_reason = "length" if truncated else "stop"
        return DIAGRAM["recovery_prediction"] + "\n"

    fake_vllm.answers[TEXT_PROMPT] = recognize
    (page,) = _post(gateway, data)

    assert _prompts(fake_vllm) == [LAYOUT_PROMPT, TEXT_PROMPT]
    blocks = _blocks(page)
    assert [b["block_label"] for b in blocks] == ["text", "image", "text", "text"]
    assert blocks[0]["block_content"] == "Native legal heading remains unchanged"
    assert blocks[2]["block_content"] == DIAGRAM["recovery_prediction"]
    assert blocks[2]["block_source"] == "ocr" and blocks[2]["block_review"]
    assert blocks[2]["block_bbox"] == blocks[1]["block_bbox"]  # the whole picture's coarse box
    assert blocks[3]["block_content"] == "Native footer remains last"
    request = page["prunedResult"]["routing"]["requests"][0]
    assert request["picture_text_blocks"] == 1
    assert "picture_text_recovery" in _codes(page)
    assert ("ocr_truncated" in _codes(page)) == truncated
    assert ("truncated" in request["issues"]) == truncated


def test_full_page_ocr_does_not_recover_pictures(gateway, fake_vllm):
    png = BytesIO()
    Image.fromarray(diagram()).save(png, format="PNG")
    with pymupdf.open() as doc:
        doc.new_page(width=300, height=160).insert_image((0, 0, 300, 160), stream=png.getvalue())
        data = doc.tobytes()
    fake_vllm.answers[LAYOUT_PROMPT] = "[{'label': 'Picture', 'bbox': [0, 0, 1000, 1000]}]"
    (page,) = _post(gateway, data)

    assert _prompts(fake_vllm) == [LAYOUT_PROMPT]
    assert page["prunedResult"]["routing"]["effective_mode"] == "full-ocr"
    assert [b["block_label"] for b in _blocks(page)] == ["image"]


def test_image_input_is_full_page_two_stage(gateway, fake_vllm):
    fake_vllm.answers[LAYOUT_PROMPT] = "[{'label': 'Text', 'bbox': [100, 100, 900, 300]}]"
    fake_vllm.answers[TEXT_PROMPT] = "Image text"
    image = Image.new("RGB", (800, 600), "white")
    response = gateway.post("/layout-parsing", json={
        "file": image_base64(image), "fileType": 1, "vlmExtraArgs": {"parsingMode": "routed"}})
    assert response.status_code == 200, response.text
    (page,) = response.json()["result"]["layoutParsingResults"]

    routing = page["prunedResult"]["routing"]
    assert routing["decision"] == "ocr" and routing["reasons"] == ["image_input_without_text_layer"]
    assert routing["effective_mode"] == "full-ocr"
    assert [(b["block_content"], b["block_bbox"]) for b in _blocks(page)] == [("Image text", [80, 60, 720, 180])]


def test_native_adapter_failure_keeps_text_only_extraction(gateway, fake_vllm, monkeypatch):
    from monkeyocr_gateway.router.native_tables import PyMuPDFTablesAdapter

    def broken(*_args):
        raise ValueError("overlapping cells")

    monkeypatch.setattr(PyMuPDFTablesAdapter, "extract", broken)
    with table_pdf() as doc:
        data = doc.tobytes()
    (page,) = _post(gateway, data)

    routing = page["prunedResult"]["routing"]
    assert "overlapping cells" in routing["native_fallback"]
    assert "native_adapter_fallback" in _codes(page) and routing["review"]
    assert all(b["block_label"] == "text" for b in _blocks(page))
    assert "Wide header" in page["markdown"]["text"]


def test_routed_is_a_valid_default_mode(fake_vllm, monkeypatch):
    monkeypatch.setenv("MOCR2_PARSING_MODE", "routed")
    with pymupdf.open() as doc:
        doc.new_page().insert_text((50, 60), "Native legal heading with enough valid characters")
        data = doc.tobytes()
    with TestClient(create_app()) as client:
        response = client.post("/layout-parsing", json={"file": base64.b64encode(data).decode(), "fileType": 0})
        assert response.status_code == 200, response.text
        page = response.json()["result"]["layoutParsingResults"][0]
        assert page["prunedResult"]["model_settings"]["parsing_mode"] == "routed"
        response = client.post("/layout-parsing", json={
            "file": base64.b64encode(data).decode(), "vlmExtraArgs": {"parsingMode": "native"}})
        assert response.status_code == 422
        assert "'routed'" in response.json()["errorMsg"]


def test_other_modes_do_not_add_routing_fields(gateway, fake_vllm):
    with pymupdf.open() as doc:
        doc.new_page().insert_text((50, 60), "Native legal heading with enough valid characters")
        data = doc.tobytes()
    fake_vllm.answers[LAYOUT_PROMPT] = "[{'label': 'Text', 'bbox': [50, 50, 950, 200]}]"
    (page,) = _post(gateway, data, vlmExtraArgs={"parsingMode": "two-stage"})
    assert "routing" not in page["prunedResult"]
    assert "page_router" not in page["prunedResult"]["model_settings"]
    assert all("block_source" not in b for b in _blocks(page))


@pytest.fixture(params=[0, 1], ids=["pdf", "image"])
def ocr_input(request):
    """Visible synthetic content with no trustworthy text layer; always full OCR."""
    if request.param == 0:
        with pymupdf.open() as doc:
            p = doc.new_page(width=300, height=300)
            p.insert_text((20, 40), "Hidden broken text layer with enough characters", render_mode=3)
            p.insert_text((20, 100), "Visible document content")
            return doc.tobytes(), 0
    return text_image_png(), 1


@pytest.mark.parametrize("truncated", [False, True])
def test_empty_ocr_output_requires_review(gateway, fake_vllm, ocr_input, truncated):
    data, file_type = ocr_input
    fake_vllm.finish_reason = "length" if truncated else "stop"
    (page,) = _post(gateway, data, file_type)
    routing = page["prunedResult"]["routing"]
    assert _blocks(page) == []
    assert routing["review"]
    assert "ocr_empty_output" in _codes(page)
    assert "empty_output" in routing["requests"][0]["issues"]
    assert ("ocr_truncated" in _codes(page)) == truncated


def test_empty_supplement_keeps_native_text_and_requires_review(gateway, fake_vllm):
    with pymupdf.open() as doc:
        p = doc.new_page(width=300, height=320)
        p.insert_text((20, 40), "Native heading with enough valid characters")
        p.insert_image((30, 100, 270, 230), stream=text_image_png())
        data = doc.tobytes()
    (page,) = _post(gateway, data)
    routing = page["prunedResult"]["routing"]
    assert routing["effective_mode"] == "native+ocr"
    assert routing["review"]
    assert {"supplemental_ocr", "ocr_empty_output"} <= set(_codes(page))
    assert [b["block_content"] for b in _blocks(page)] == ["Native heading with enough valid characters"]
    assert all(b["block_source"] == "native" for b in _blocks(page))
    assert "empty_output" in routing["requests"][0]["issues"]


@pytest.mark.parametrize("label", ["Picture", "Table"])
def test_structural_ocr_blocks_propagate_prediction_review(gateway, fake_vllm, ocr_input, label):
    """Source convert_records marks every block when any picture/table needs review."""
    data, file_type = ocr_input
    fake_vllm.answers[LAYOUT_PROMPT] = repr([
        {"label": "Text", "bbox": [0, 0, 1000, 200]},
        {"label": label, "bbox": [0, 200, 1000, 1000]},
    ])
    fake_vllm.answers[TEXT_PROMPT] = "Recognized heading"
    fake_vllm.answers[TABLE_PROMPT] = "<fcel>Value<nl>"
    (page,) = _post(gateway, data, file_type)
    assert page["prunedResult"]["routing"]["review"]
    assert all(b["block_review"] for b in _blocks(page))
    code = "picture_requires_review" if label == "Picture" else "table_requires_review"
    assert code in _codes(page)
    assert _blocks(page)[0]["block_content"] == "Recognized heading"
    assert _prompts(fake_vllm).count(TEXT_PROMPT) == 1  # no full-page picture recovery


@pytest.mark.parametrize("label", ["Text", "Table"])
def test_ocr_script_checks_apply_to_images_and_pdf(gateway, fake_vllm, ocr_input, label):
    data, file_type = ocr_input
    raw = "QUYẾT ĐỊNH СỘNG HÒА α"
    fake_vllm.answers[LAYOUT_PROMPT] = repr([{"label": label, "bbox": [0, 0, 1000, 1000]}])
    fake_vllm.answers[TEXT_PROMPT] = raw
    fake_vllm.answers[TABLE_PROMPT] = f"<fcel>{raw}<nl>"
    (page,) = _post(gateway, data, file_type)
    assert page["prunedResult"]["routing"]["review"]
    assert {"unexpected_script", "mixed_script_word"} <= set(_codes(page))
    assert raw in _blocks(page)[0]["block_content"]  # flag, never repair predictions


@pytest.mark.parametrize("stage", ["layout", "recognition"])
def test_truncated_ocr_keeps_content_and_requires_review(gateway, fake_vllm, ocr_input, stage):
    data, file_type = ocr_input

    def layout(_image):
        fake_vllm.finish_reason = "length" if stage == "layout" else "stop"
        return "[{'label': 'Text', 'bbox': [0, 0, 1000, 1000]}]"

    def recognize(_image):
        fake_vllm.finish_reason = "length" if stage == "recognition" else "stop"
        return "Partial transcription"

    fake_vllm.answers[LAYOUT_PROMPT] = layout
    fake_vllm.answers[TEXT_PROMPT] = recognize
    (page,) = _post(gateway, data, file_type)
    assert page["prunedResult"]["routing"]["review"]
    assert "ocr_truncated" in _codes(page)
    assert "truncated" in page["prunedResult"]["routing"]["requests"][0]["issues"]
    assert [(b["block_content"], b["block_review"]) for b in _blocks(page)] == [
        ("Partial transcription", True)]

    # Quality evidence belongs to a request, not shared inference-client state.
    fake_vllm.finish_reason = "stop"
    fake_vllm.answers[LAYOUT_PROMPT] = "[{'label': 'Text', 'bbox': [0, 0, 1000, 1000]}]"
    fake_vllm.answers[TEXT_PROMPT] = "Complete transcription"
    (complete,) = _post(gateway, data, file_type)
    assert not complete["prunedResult"]["routing"]["review"]
    assert not _blocks(complete)[0]["block_review"]
    assert _codes(complete) == []
