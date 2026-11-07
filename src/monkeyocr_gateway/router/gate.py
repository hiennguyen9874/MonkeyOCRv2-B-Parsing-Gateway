"""Conservative, inspectable native quality gate; no linguistic correction.

Ported verbatim from legal-vi-ocr `src/legal_vi_ocr/router.py` at 6e7ad59.
"""

from dataclasses import asdict, dataclass
import unicodedata

import numpy as np
import pymupdf

from .geometry import background, gray, text_mask
from .profiler import profile_page


@dataclass(frozen=True)
class GateConfig:
    min_characters: int = 20
    min_valid_diacritic_ratio: float = 0.9
    max_bad_glyph_ratio: float = 0.01
    max_invisible_ratio: float = 0.1
    max_outside_ratio: float = 0.05
    max_overlap_ratio: float = 0.1
    min_image_area_ratio: float = 0.04
    min_missing_ink_pixels: int = 80
    min_missing_ink_ratio: float = 0.002
    near_blank_ink_ratio: float = 0.00005
    analysis_dpi: int = 100
    faint_contrast: int = 8
    small_crop_padding_pt: int = 12


def has_text_lines(mask):
    """Conservative projection heuristic, not a signature/seal detector.

    Require separated strokes across a wide, shallow band. Try both axes so a
    rotated small textual image is not silently treated as empty.
    """
    for ink in (mask, mask.T):
        rows = ink.sum(1) >= 2
        edges = np.flatnonzero(np.diff(np.r_[False, rows, False]))
        for start, end in zip(edges[::2], edges[1::2]):
            band = ink[start:end]
            columns = np.flatnonzero(band.any(0))
            if (len(columns) >= 8 and band.sum() >= 20
                    and columns[-1] - columns[0] >= 2.5 * (end - start)
                    and np.count_nonzero(np.diff(columns) > 1) >= 2):
                return True
    return False


def merged_regions(rectangles):
    """Union connected overlapping placements before classifying/requesting them."""
    regions = []
    for rect in rectangles:
        rect = pymupdf.Rect(rect)
        changed = True
        while changed:
            changed = False
            for i, old in enumerate(regions):
                if (old & rect).get_area() > 0:
                    rect |= regions.pop(i)
                    changed = True
                    break
        regions.append(rect)
    return sorted(regions, key=lambda r: -r.get_area())


def word_boxes(page):
    return [pymupdf.Rect(w[:4]) for w in page.get_text("words") if w[4].strip()]


def pixel_box(rect, matrix, width, height, bounds):
    r = pymupdf.Rect(rect) * matrix
    sx, sy = width / bounds.width, height / bounds.height
    return (max(0, min(width, int(r.x0 * sx))), max(0, min(height, int(r.y0 * sy))),
            max(0, min(width, int(np.ceil(r.x1 * sx)))), max(0, min(height, int(np.ceil(r.y1 * sy)))))


def assess(page, config=GateConfig()):
    """Return JSON signals/reasons and image regions in *unrotated* PDF points.

    Visibility is a heuristic: render mode, opacity, white text, clipping and later
    image/fill bounding boxes. Bounding boxes cannot prove arbitrary path occlusion.
    """
    profile = profile_page(page, "", 1)
    count = profile["character_count"]
    bounds = page.rect * page.derotation_matrix
    traces, log = page.get_texttrace(), page.get_bboxlog()
    total = invisible = outside = overlap = 0
    seen = {}
    for span in traces:
        hidden = span["type"] == 3 or span["opacity"] < 0.1 or min(span["color"]) > 0.95
        later = [pymupdf.Rect(box) for kind, box in log[span["seqno"] + 1:]
                 if kind in ("fill-image", "fill-path")]
        for code, glyph, origin, box in span["chars"]:
            ch = chr(code)
            if ch.isspace():
                continue
            total += 1
            rect = pymupdf.Rect(box)
            area = rect.get_area()
            occluded = area > 0 and any((rect & r).get_area() > 0.95 * area for r in later)
            invisible += hidden or occluded
            outside += not area or (rect & bounds).get_area() < 0.8 * area
            # Near-identical glyph origins/boxes expose duplicated overprinted layers,
            # without treating adjacent letters or separate Vietnamese marks as overlap.
            key = (round(origin[0]), round(origin[1]))
            duplicates = seen.setdefault(key, [])
            if area > 0:
                overlap += any((rect & r).get_area() > 0.85 * min(area, r.get_area()) for r in duplicates)
            duplicates.append(rect)
    pix = page.get_pixmap(dpi=config.analysis_dpi, colorspace=pymupdf.csRGB, alpha=False)
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    dark = rgb.min(2) < 160
    lum = gray(rgb)
    _, paper = background(lum, max(8, round(0.21 * config.analysis_dpi)))
    faint = paper - lum >= config.faint_contrast
    ink = text_mask(dark, rgb, config.analysis_dpi)
    # Absolute chroma alone admits pale red stamp edges as gray text. Require
    # near-neutral ink relative to its contrast as well (review coloured marks).
    chroma = rgb.max(2).astype(np.int16) - rgb.min(2).astype(np.int16)
    neutral = chroma <= np.maximum(12, 0.35 * (255 - rgb.min(2).astype(float)))
    faint_ink = text_mask(faint, rgb, config.analysis_dpi) & neutral
    ink &= neutral
    uncovered = ink | faint_ink
    # Keep coloured/unknown marks as pictures, rather than silently dropping them.
    uncovered_marks = dark | faint
    for box in word_boxes(page):
        x0, y0, x1, y1 = pixel_box(box + (-1, -1, 1, 1), page.rotation_matrix,
                                 pix.width, pix.height, page.rect)
        uncovered[y0:y1, x0:x1] = False
        uncovered_marks[y0:y1, x0:x1] = False
    regions, retained, candidates = [], [], []
    placements = [pymupdf.Rect(i["bbox"]) & bounds for i in page.get_image_info()]
    for rect in merged_regions([r for r in placements if r.get_area() > 0]):
        x0, y0, x1, y1 = pixel_box(rect, page.rotation_matrix, pix.width, pix.height, page.rect)
        missing = int(uncovered[y0:y1, x0:x1].sum())
        marks = int(uncovered_marks[y0:y1, x0:x1].sum())
        if not marks:
            continue
        fraction = missing / max(1, (x1 - x0) * (y1 - y0))
        small = rect.get_area() < config.min_image_area_ratio * bounds.get_area()
        lines = has_text_lines(uncovered[y0:y1, x0:x1])
        recognize = lines or (not small and missing >= config.min_missing_ink_pixels
                              and fraction >= config.min_missing_ink_ratio)
        candidates.append({"bbox": list(rect), "small": small, "missing_ink_pixels": missing,
                           "uncovered_mark_pixels": marks, "text_lines": lines,
                           "action": "ocr" if recognize else "picture"})
        if recognize:
            regions.append(list(rect))
        else:
            retained.append({"bbox": list(rect), "reason": "unrecognized_small_image" if small
                             else "unrecognized_image"})
    def ratio(n):
        return n / max(1, total)

    signals = {k: profile[k] for k in ("character_count", "valid_vietnamese_diacritic_ratio",
                                       "invalid_diacritic_marks", "bad_glyph_count")}
    signals.update(traced_characters=total, invisible_ratio=ratio(invisible),
                   outside_ratio=ratio(outside), overlap_ratio=ratio(overlap),
                   text_ink_ratio=float(ink.mean()), faint_ink_ratio=float(faint_ink.mean()),
                   mark_ink_ratio=float((dark | faint).mean()),
                   image_regions=regions, image_candidates=candidates)
    reasons = []
    if count < config.min_characters:
        reasons.append("insufficient_characters")
    valid = signals["valid_vietnamese_diacritic_ratio"]
    if valid is not None and valid < config.min_valid_diacritic_ratio:
        reasons.append("invalid_vietnamese_diacritics")
    raw = unicodedata.normalize("NFC", page.get_text())
    bad = sum(ch == "\ufffd" or unicodedata.category(ch) in ("Co", "Cs", "Cn")
              or (unicodedata.category(ch) == "Cc" and not ch.isspace()) for ch in raw)
    signals["bad_glyph_count"] = bad
    if bad / max(1, count) > config.max_bad_glyph_ratio:
        reasons.append("bad_glyphs")
    for metric, maximum, reason in (("invisible_ratio", config.max_invisible_ratio, "invisible_text"),
                                    ("outside_ratio", config.max_outside_ratio, "outside_page"),
                                    ("overlap_ratio", config.max_overlap_ratio, "overlapping_text")):
        if signals[metric] > maximum:
            reasons.append(reason)
    if count and not total:
        reasons.append("missing_visibility_evidence")
    decision = "ocr" if reasons else "native"
    if not reasons:
        reasons = ["native_quality_gate_passed"]
        if regions:
            reasons.append("uncovered_image_ink")
    low_ink = count == 0 and signals["text_ink_ratio"] < config.near_blank_ink_ratio
    # Any measurable faint/local-contrast content gets OCR; A10 is intentionally
    # recognized rather than assumed empty. Only genuinely ink-free pages skip it.
    near_blank = low_ink and not np.any(dark | faint)
    review = []
    if low_ink:
        review.append({"code": "near_blank_ocr_suppressed" if near_blank else "low_contrast_ocr_required"})
    return {"decision": decision, "reasons": reasons,
            "signals": signals, "thresholds": asdict(config), "near_blank": near_blank,
            "effective_mode": "near-blank-review" if near_blank else (
                "full-ocr" if decision == "ocr" else "native+ocr" if regions else "native-only"),
            "review_reasons": review,
            "retained_regions": retained if decision == "native" else [],
            "supplement_regions": regions if decision == "native" else [], "requests": []}
