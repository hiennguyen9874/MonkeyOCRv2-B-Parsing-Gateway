"""Review-only routing annotations and spatial anchors; never repair predictions.

Ported from legal-vi-ocr `src/legal_vi_ocr/routing_review.py` at 6e7ad59 (`flag`, `picture`,
`spatial_merge`, `annotate`). Adapted to the gateway's routed Khối (see `routed.py`) instead of
canonical blocks; the rules are the same. Gateway adaptations:

- no `ocr_pending` / `ocr_failed` flags: a failed inference fails the whole HTTP request;
- truncation and empty predictions carry request-local issues instead of canonical page quality;
- pictures explicitly carry `picture_requires_review` as well as their engine review flag;
- `spatial_merge` takes the page rotation directly, so retained pictures on rotated pages anchor
  correctly even without OCR requests;
- `spatial_merge` leaves a page without native Khối as it is: full-page OCR keeps Monkey's
  reading order, as in two-stage mode, instead of being sorted top-to-bottom/left-to-right.
"""

import unicodedata

import pymupdf

MODES = ("native-only", "native+ocr", "full-ocr", "near-blank-review")


def flag(routing, code, **evidence):
    reason = {"code": code, **evidence}
    if reason not in routing["review_reasons"]:
        routing["review_reasons"].append(reason)


def picture(bbox, config=None):
    """A retained Picture Khối: kept for review, never recognized."""
    return {"bbox": list(bbox), "label": "Picture", "content": "", "raw_text": "",
            "source": "routing", "review": True, "config": dict(config or {})}


def is_addition(block):
    """Supplemental OCR and retained pictures are placed at native anchors."""
    config = block["config"]
    return "routed_request" in config or bool(config.get("routing_picture"))


def spatial_merge(page):
    """Insert supplements at native anchors; never sort the native sequence.

    Work in rendered-page axes, including PDF /Rotate. Sorting only supplements
    within each anchor avoids interleaving two-column native headers.
    """
    matrix = pymupdf.Matrix(*page.get("rotation_matrix", (1, 0, 0, 1, 0, 0)))
    native, additions = [], []
    for block in page["blocks"]:
        (additions if is_addition(block) else native).append(block)
    if not native:
        return
    anchors = [[] for _ in range(len(native) + 1)]
    for block in additions:
        rect = pymupdf.Rect(block["bbox"]) * matrix
        at = len(native)
        for i, source in enumerate(native):
            other = pymupdf.Rect(source["bbox"]) * matrix
            if other.y0 >= rect.y1 - 1 and min(other.x1, rect.x1) > max(other.x0, rect.x0):
                at = i
                break
        anchors[at].append(block)

    def position(block):
        r = pymupdf.Rect(block["bbox"]) * matrix
        return r.y0, r.x0

    merged = []
    for i, block in enumerate(native):
        merged.extend(sorted(anchors[i], key=position))
        merged.append(block)
    merged.extend(sorted(anchors[-1], key=position))
    page["blocks"] = merged


def annotate(page):
    """Structured review evidence is independent of successful task completion."""
    routing = page["routing"]
    if routing.get("native_fallback"):
        flag(routing, "native_adapter_fallback", error=routing["native_fallback"])
    for request in routing["requests"]:
        name = request["id"]
        if request["region"] is not None:
            flag(routing, "supplemental_ocr", request=name)
        for issue, code in (("empty_output", "ocr_empty_output"), ("truncated", "ocr_truncated")):
            if issue in request.get("issues", []):
                flag(routing, code, request=name)
        if "empty_picture_text_recovery" in request.get("issues", []):
            flag(routing, "empty_picture_text_recovery", request=name)
        if request.get("dropped_spatial_duplicates"):
            flag(routing, "spatial_duplicates_dropped", request=name,
                 count=request["dropped_spatial_duplicates"])
    for i, block in enumerate(page["blocks"]):
        config = block["config"]
        if "picture_text_source" in config:
            flag(routing, "picture_text_recovery", block=i, source=config["picture_text_source"],
                 request=config.get("routed_request"))
        if "table_fallback" in config:
            # Record even benign rejections (e.g. a one-row prose box).
            note = config["table_fallback"]
            if note not in routing.setdefault("table_fallbacks", []):
                routing["table_fallbacks"].append(note)
            if note.get("review"):
                flag(routing, "native_table_fallback", **{"table": note})
        if block["label"] == "Table":
            flag(routing, "table_requires_review", block=i)
        if block["label"] == "Picture":
            flag(routing, "picture_requires_review", block=i)
        if block["review"]:
            flag(routing, "block_requires_review", block=i)
        # Inspect plain text (table cells included), without modifying it. Greek and
        # Cyrillic are unusual in this corpus, not necessarily transcription errors.
        scripts = set()
        mixed = False
        for word in block["raw_text"].split():
            word_scripts = set()
            for ch in word:
                if not ch.isalpha():
                    continue
                name = unicodedata.name(ch, "")
                script = next((s for s in ("LATIN", "CYRILLIC", "GREEK") if s in name), None)
                if script:
                    word_scripts.add(script)
            scripts |= word_scripts & {"CYRILLIC", "GREEK"}
            mixed |= len(word_scripts) > 1
        if scripts:
            flag(routing, "unexpected_script", block=i, scripts=sorted(scripts))
        if mixed:
            flag(routing, "mixed_script_word", block=i)
    if page["review"] and not routing["review_reasons"]:
        flag(routing, "engine_requires_review")
    # Very sparse dark ink can still be real faint text: flag, never delete it.
    signals = routing["signals"]
    if ("text_ink_ratio" in signals
            and signals["text_ink_ratio"] < routing["thresholds"]["near_blank_ink_ratio"]
            and routing["decision"] == "ocr" and any(b["raw_text"].strip() for b in page["blocks"])):
        flag(routing, "possible_low_ink_hallucination")
    page["review"] |= bool(routing["review_reasons"])
