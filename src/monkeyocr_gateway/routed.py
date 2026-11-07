"""Chế độ định tuyến theo trang (routed mode): native-first, per page (ADR-0007).

The gateway counterpart of legal-vi-ocr `src/legal_vi_ocr/routed_run.py` at 6e7ad59
(`docs/page-router-architecture.md` there). Per PDF page:

1. `router.gate.assess` decides from the embedded text layer (Lớp chữ có sẵn): `native` or `ocr`.
2. `native`: Khối come from PyMuPDF (text + post-processed tables). Image ink the text
   layer does not cover becomes a supplemental OCR request (crop, native words masked
   white, white padding when small) or a retained Picture for review.
3. `ocr`: the whole page goes through Chế độ hai giai đoạn, as two-stage mode does.
4. A page with no text and no measurable ink keeps one empty Picture for review.
5. OCR Khối are mapped back to the source page, spatial duplicates of native words are
   dropped, supplements are inserted at native anchors, and review flags are added.

What the batch runner does and the gateway does not: diagnostic manifests and image
hash checks (the gateway renders the page itself), prepare-only / replay of recorded
runs (tests use the fake inference server instead), on-disk artifacts, and per-page
error status (any failed inference call fails the request, as in the other modes).

Internally Khối carry bboxes in unrotated PDF points relative to the CropBox, the
coordinate space of PyMuPDF text extraction. `to_khoi` converts them to the rendered
page pixels every Kết quả trang uses.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import threading
from dataclasses import asdict

import numpy as np
import pymupdf
from PIL import Image

from .document_loader import DocumentError, PDF_RENDER_SCALE
from .page_parser import _chat_with_repeat_retry, gather_or_cancel, parse_page
from .router import SOURCE_REVISION
from .router.gate import assess, GateConfig, pixel_box, word_boxes
from .router.native import PyMuPDFAdapter
from .router.native_tables import PyMuPDFTablesAdapter
from .router.picture_recovery import candidates as picture_candidates
from .router.picture_recovery import POLICY as PICTURE_RECOVERY_POLICY
from .router.review import annotate, flag, picture, spatial_merge
from .router.views import table_html
from .upstream import ALL_PROMPT, RECOGNITION_MAX_TOKENS
from .vllm_client import GenerationOptions, VLLMClient

# MuPDF is not thread-safe; every PyMuPDF call on a document takes this lock.
MUPDF_LOCK = threading.RLock()
# Pages are rendered by document_loader at Upstream's 200 DPI.
RENDER_DPI = round(PDF_RENDER_SCALE * 72)
# The OCR branch is Monkey two-stage (legal-vi-ocr ADR-0003).
OCR_PARSING_MODE = "two-stage"
GATE_CONFIG = GateConfig()

PAGE_ROUTER_SETTINGS = {
    "source": SOURCE_REVISION,
    "native_engine": "pymupdf-tables",
    "ocr_parsing_mode": OCR_PARSING_MODE,
    "supplement_policy": "mask native words; review all supplements",
    "supplemental_picture_text_recovery": PICTURE_RECOVERY_POLICY,
    "reading_order": "supplements at spatial native anchors; native relative order unchanged",
}


class NativeDocument:
    """The PDF read with PyMuPDF for its text layer; pages are planned on demand.

    Each plan opens its own copy of the document: on a rotated page, PyMuPDF
    `Page.find_tables()` rewrites the page (content stream, MediaBox, which also
    drops the CropBox), and the table adapter runs it on the previous page to
    detect a continued table. A fresh copy keeps every page's geometry as the PDF has it.
    """

    def __init__(self, data: bytes):
        try:
            with MUPDF_LOCK, pymupdf.open(stream=data, filetype="pdf") as doc:
                self.page_count = len(doc)
        except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
            raise DocumentError(422, f"file is not a readable PDF: {exc}") from None
        self._data = data
        # The canonical contract of the native adapters needs a source path and hash.
        self._file = {"path": "input.pdf", "sha256": hashlib.sha256(data).hexdigest()}

    def plan(self, index: int) -> dict:
        with MUPDF_LOCK, pymupdf.open(stream=self._data, filetype="pdf") as doc:
            return plan_page(doc[index], self._file, index + 1)

    def close(self) -> None:
        self._data = b""


def native_block(block: dict) -> dict:
    """Canonical native block (text or table) -> routed Khối with a Nhãn Monkey."""
    config = {
        key: block["provenance"]["config"][key]
        for key in ("table_fallback", "table_postprocess")
        if key in block["provenance"]["config"]
    }
    routed = {
        "bbox": list(block["bbox"]),
        "label": "Text",
        "content": block["raw_text"],
        "raw_text": block["raw_text"],
        "source": "native",
        "review": block["quality"]["review"],
        "config": config,
    }
    if block["type"] == "table":
        # Cell texts for review checks; HTML for the Paddle table Khối.
        routed.update(
            label="Table",
            content=table_html(block["table"]),
            format="html",
            raw_text=" ".join(c["text"] for c in block["table"]["cells"]),
        )
    return routed


def plan_page(pdf_page, file: dict, number: int) -> dict:
    """Gate the page and extract its native Khối; CPU only, no inference.

    Mirrors the per-page preparation of routed_run.run: the native extraction always
    runs (it is the routing evidence), the table adapter only when the gate passes.
    Page geometry is read first, as the table adapter may rewrite a rotated page.
    """
    geometry = {
        "rect": [pdf_page.rect.width, pdf_page.rect.height],
        "bounds": list(pymupdf.Rect(pdf_page.rect * pdf_page.derotation_matrix).normalize()),
        "rotation_matrix": list(pdf_page.rotation_matrix),
        "derotation_matrix": list(pdf_page.derotation_matrix),
        "word_boxes": [list(box) for box in word_boxes(pdf_page)],
    }
    routing = assess(pdf_page, GATE_CONFIG)
    page = PyMuPDFAdapter().extract(pdf_page, file, number)
    if routing["decision"] == "native" and pdf_page.rotation:
        # Gateway difference: find_tables() works on the rotated page while the adapter
        # compares with unrotated text, so its tables would be rejected or misplaced.
        # Keep the text-only extraction and say so.
        routing["native_tables_skipped"] = "rotated_page"
        blocks = [native_block(block) for block in page["blocks"]]
    elif routing["decision"] == "native":
        try:
            page = PyMuPDFTablesAdapter().extract(pdf_page, file, number)
        except Exception as exc:  # noqa: BLE001 - recorded, text-only extraction kept
            routing["native_fallback"] = f"{type(exc).__name__}: {exc}"
            page["quality"]["review"] = True
        blocks = [native_block(block) for block in page["blocks"]]
    else:
        blocks = []
    review = page["quality"]["review"]
    regions: list = []
    if routing["near_blank"]:
        routing["reasons"].append("near_blank_retained_without_recognition")
        review = True
        blocks = [picture(geometry["bounds"])]
    else:
        regions = [None] if routing["decision"] == "ocr" else list(routing["supplement_regions"])
    for region in routing["retained_regions"]:
        blocks.append(picture(region["bbox"], {"routing_picture": region["reason"]}))
        flag(routing, region["reason"], bbox=region["bbox"])
    routing["page_type"] = "scan" if not routing["signals"]["character_count"] else "native"
    routing["coordinate_unit"] = "pt"
    if not any(regions):
        geometry["word_boxes"] = []  # only supplements mask native words
    return {"routing": routing, "blocks": blocks, "review": review, "regions": regions, **geometry}


def prepare_request(
    image: Image.Image, plan: dict, index: int, region
) -> tuple[dict, Image.Image | None]:
    """The OCR input of one request and the mapping of its points back to the source.

    Full-page requests send the rendered page unchanged. Supplements crop the region,
    mask native words (1 pt margin) white and pad small crops with white, extending
    the affine mapping by exactly the padding so source boxes stay reversible.
    Returns (request, None) when the region covers no pixel.
    """
    width, height = image.size
    rect = pymupdf.Rect(0, 0, *plan["rect"])
    rotation = pymupdf.Matrix(*plan["rotation_matrix"])
    masked = region is not None
    if region is None:
        box = (0, 0, width, height)
    else:
        box = pixel_box(region, rotation, width, height, rect)
    x0, y0, x1, y1 = box
    visual = pymupdf.Rect(
        x0 * rect.width / width,
        y0 * rect.height / height,
        x1 * rect.width / width,
        y1 * rect.height / height,
    )
    request = {
        "id": f"request-{index:04d}",
        "region": region,
        "masked_native_words": masked,
        "padding_pixels": 0,
        "image_size": [x1 - x0, y1 - y0],
    }
    if x1 <= x0 or y1 <= y0:
        request["visual_bbox"] = list(visual)
        return request, None
    if region is None:
        ocr_image = image
    else:
        crop = np.asarray(image)[y0:y1, x0:x1].copy()
        for word in plan["word_boxes"]:
            a, b, c, d = pixel_box(
                pymupdf.Rect(word) + (-1, -1, 1, 1), rotation, width, height, rect
            )
            a, b, c, d = max(a, x0), max(b, y0), min(c, x1), min(d, y1)
            if a < c and b < d:
                crop[b - y0 : d - y0, a - x0 : c - x0] = 255
        if pymupdf.Rect(region).get_area() < GATE_CONFIG.min_image_area_ratio * rect.get_area():
            padding = max(1, round(GATE_CONFIG.small_crop_padding_pt * RENDER_DPI / 72))
            crop = np.pad(
                crop, ((padding, padding), (padding, padding), (0, 0)), constant_values=255
            )
            px, py = padding * rect.width / width, padding * rect.height / height
            visual += (-px, -py, px, py)
            request["padding_pixels"] = padding
        ocr_image = Image.fromarray(crop)
        request["image_size"] = list(ocr_image.size)
    request["visual_bbox"] = list(visual)
    return request, ocr_image


async def recover_pictures(
    ocr_image: Image.Image,
    blocks: list[dict],
    vllm: VLLMClient,
    options: GenerationOptions,
    repeat_retries: int,
    issues: list[str] | None = None,
) -> list[dict]:
    """Recognize text-like Picture Khối of a supplement with Upstream's Text prompt.

    The original Picture stays; each recovered Khối has the whole picture's box.
    legal-vi-ocr monkey_two_stage_run.recover_pictures, over the gateway client.
    """
    pictures = [(i, block["bbox"]) for i, block in enumerate(blocks) if block["label"] == "Picture"]
    if not pictures:
        return []
    recognized = [block["bbox"] for block in blocks if block["content"].strip()]
    found = await asyncio.to_thread(
        picture_candidates, np.asarray(ocr_image), pictures, recognized, RENDER_DPI
    )
    raws = await gather_or_cancel(
        [
            _chat_with_repeat_retry(
                vllm,
                Image.fromarray(candidate["image"]),
                ALL_PROMPT["Text"],
                options,
                repeat_retries,
                RECOGNITION_MAX_TOKENS,
                issues=issues,
            )
            for candidate in found
        ]
    )
    reason = "text_like_ink_in_unrecognized_picture"
    return [
        {
            "bbox": candidate["bbox"],
            "label": "Text",
            "content": (raw or "").strip(),
            "config": {
                "picture_text_recovery": PICTURE_RECOVERY_POLICY,
                "picture_text_source": {
                    "picture_block": candidate["picture_block"],
                    "bbox": candidate["bbox"],
                    "reason": reason,
                },
            },
        }
        for candidate, raw in zip(found, raws, strict=True)
    ]


async def parse_ocr_request(
    image: Image.Image,
    request: dict,
    vllm: VLLMClient,
    options: GenerationOptions,
    repeat_retries: int,
) -> list[dict]:
    """Shared PDF/image OCR quality conversion, after source convert_records at 6e7ad59.

    The source marks all blocks in a prediction for review when it is truncated or
    contains tables/pictures. Empty predictions still carry request-level evidence.
    Keep this evidence local to the request, including supplemental picture recovery.
    """
    issues = request.setdefault("issues", [])
    blocks = await parse_page(image, OCR_PARSING_MODE, vllm, options, repeat_retries, issues=issues)
    request["blocks"] = len(blocks)
    if not blocks:
        issues.append("empty_output")
    for label, code in (("Table", "table_review"), ("Picture", "picture_review")):
        if any(block["label"] == label for block in blocks):
            issues.append(code)
    if request["masked_native_words"]:
        # Only supplements opt into picture text recovery, as in the batch runner.
        recovered = await recover_pictures(image, blocks, vllm, options, repeat_retries, issues)
        request["picture_text_blocks"] = len(recovered)
        if recovered:
            issues.append("picture_text_recovery_review")
        if any(not block["content"] for block in recovered):
            issues.append("empty_picture_text_recovery")
        blocks = blocks + recovered
    return [
        {
            **block,
            "raw_text": block["content"],
            "source": "ocr",
            "review": bool(issues),
            "config": dict(block.get("config", {})),
        }
        for block in blocks
    ]


def merge_prediction(page: dict, plan: dict, blocks: list[dict], request: dict) -> None:
    """Map OCR Khối from request pixels to source points; keep label and content.

    routed_run.merge_prediction: supplements never trust a generative model not to
    repeat adjacent native text, so Khối with text more than half covered by native
    word boxes are dropped as spatial duplicates; strings are never compared.
    """
    visual = pymupdf.Rect(request["visual_bbox"])
    width, height = request["image_size"]
    sx, sy = visual.width / width, visual.height / height
    matrix = pymupdf.Matrix(*plan["derotation_matrix"])
    bounds = pymupdf.Rect(plan["bounds"])
    supplement = request["region"] is not None
    native_words = [pymupdf.Rect(w) for w in plan["word_boxes"]] if supplement else []
    additions, dropped = [], 0
    for block in blocks:
        x0, y0, x1, y1 = block["bbox"]
        r = pymupdf.Rect(x0 * sx, y0 * sy, x1 * sx, y1 * sy)
        r += (visual.x0, visual.y0, visual.x0, visual.y0)
        r = (r * matrix).normalize()
        if supplement and block["content"].strip():
            covered = sum((r & word).get_area() for word in native_words)
            if covered > 0.5 * max(r.get_area(), 0.001):
                dropped += 1
                continue
        # Padded request boxes may include white pixels outside the source page.
        clipped = r & bounds
        if clipped.get_area() <= 0:
            dropped += 1
            continue
        config = {
            "routed_request": request["id"],
            "source_visual_bbox": request["visual_bbox"],
            "masked_native_words": request["masked_native_words"],
            **block.get("config", {}),
        }
        additions.append(
            {
                "bbox": list(clipped),
                "label": block["label"],
                "content": block["content"],
                "raw_text": block["content"],
                "source": "ocr",
                "review": block.get("review", False)
                or supplement
                or "picture_text_source" in config,
                "config": config,
            }
        )
    page["blocks"].extend(additions)
    page["review"] |= supplement
    request["dropped_spatial_duplicates"] = dropped
    if dropped:
        page["review"] = True


def to_khoi(block: dict, plan: dict, size: tuple[int, int]) -> dict:
    """Routed Khối (unrotated points) -> page_parser Khối (rendered page pixels)."""
    width, height = size
    x0, y0, x1, y1 = pixel_box(
        block["bbox"],
        pymupdf.Matrix(*plan["rotation_matrix"]),
        width,
        height,
        pymupdf.Rect(0, 0, *plan["rect"]),
    )
    # Picture crops need at least one pixel.
    x0, y0 = min(x0, width - 1), min(y0, height - 1)
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    khoi = {
        "bbox": [x0, y0, x1, y1],
        "label": block["label"],
        "content": block["content"],
        "source": block["source"],
        "review": block["review"],
    }
    if "format" in block:
        khoi["format"] = block["format"]
    return khoi


async def parse_routed_page(
    image: Image.Image,
    plan: dict,
    vllm: VLLMClient,
    options: GenerationOptions,
    repeat_retries: int = 0,
) -> tuple[list[dict], dict]:
    """Run the OCR requests of a planned PDF page and merge them with its native Khối.

    Returns (Khối in reading order, in page pixels, as page_parser returns them; routing).
    """
    plan = copy.deepcopy(plan)
    routing = plan["routing"]
    page = {
        "blocks": plan["blocks"],
        "routing": routing,
        "review": plan["review"],
        "rotation_matrix": plan["rotation_matrix"],
    }

    async def run(index: int, region):
        request, ocr_image = await asyncio.to_thread(prepare_request, image, plan, index, region)
        if ocr_image is None:
            request["issues"] = ["empty_region"]
            return request, []
        blocks = await parse_ocr_request(ocr_image, request, vllm, options, repeat_retries)
        return request, blocks

    results = await gather_or_cancel(
        [run(index, region) for index, region in enumerate(plan["regions"], 1)]
    )
    for request, blocks in results:
        routing["requests"].append(request)
        if blocks:
            merge_prediction(page, plan, blocks, request)
        elif "empty_region" in request.get("issues", []):
            flag(routing, "empty_supplement_region", request=request["id"], bbox=request["region"])
    spatial_merge(page)
    annotate(page)
    routing["review"] = page["review"]
    return [to_khoi(block, plan, image.size) for block in page["blocks"]], routing


async def parse_routed_image(
    image: Image.Image,
    vllm: VLLMClient,
    options: GenerationOptions,
    repeat_retries: int = 0,
) -> tuple[list[dict], dict]:
    """An image input has no text layer: the whole page goes through two-stage OCR."""
    request = {
        "id": "request-0001",
        "region": None,
        "masked_native_words": False,
        "padding_pixels": 0,
        "image_size": list(image.size),
    }
    blocks = await parse_ocr_request(image, request, vllm, options, repeat_retries)
    routing = {
        "decision": "ocr",
        "reasons": ["image_input_without_text_layer"],
        "signals": {},
        "thresholds": asdict(GATE_CONFIG),
        "near_blank": False,
        "effective_mode": "full-ocr",
        "review_reasons": [],
        "retained_regions": [],
        "supplement_regions": [],
        "requests": [request],
        "page_type": "scan",
        "coordinate_unit": "pt",
    }
    page = {"blocks": blocks, "routing": routing, "review": False}
    annotate(page)
    routing["review"] = page["review"]
    return blocks, routing
