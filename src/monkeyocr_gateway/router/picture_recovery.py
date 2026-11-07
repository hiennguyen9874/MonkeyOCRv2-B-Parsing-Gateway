"""Reviewed Text-prompt fallback for textual pictures in supplemental OCR only.

This is not a seal detector or an accuracy/coverage certificate. Keep the original
Picture and records; the added text has only the picture's coarse source box.

Ported from legal-vi-ocr `src/legal_vi_ocr/picture_recovery.py` at 6e7ad59: `POLICY`,
`MIN_EXTENT_PT` and `candidates` verbatim. `pending_for_prediction` guards replay of
recorded runs, which the gateway does not have, and is not ported.
"""

import numpy as np
import pymupdf

from .gate import has_text_lines
from .geometry import background, gray, text_mask

POLICY = "textual-picture-v3"
MIN_EXTENT_PT = 144  # substantial diagrams, not a new small seal/signature reader


def candidates(rgb, pictures, recognized, dpi):
    """Return pixel-aligned picture boxes with unrecognized text-like strokes.

    Both polarities matter: diagram headings can be white on coloured backgrounds.
    Mask already recognized boxes, not strings, to avoid re-reading their labels.
    """
    height, width = rgb.shape[:2]
    bounds = pymupdf.Rect(0, 0, width, height)
    result = []
    covered_boxes = list(recognized)
    for picture_index, bbox in pictures:
        rect = (pymupdf.Rect(bbox) & bounds).irect
        if rect.is_empty or min(rect.width, rect.height) * 72 / dpi < MIN_EXTENT_PT:
            # Keep the original reviewed Picture. This narrowly scoped fallback
            # is not a replacement for primary all-size OCR or mark handling.
            continue
        crop = rgb[rect.y0:rect.y1, rect.x0:rect.x1].copy()
        for box in covered_boxes:
            covered = (pymupdf.Rect(box) & rect).irect
            if not covered.is_empty:
                crop[covered.y0-rect.y0:covered.y1-rect.y0,
                     covered.x0-rect.x0:covered.x1-rect.x0] = 255
        # Downsample analysis only; recognition uses the full-resolution crop.
        step = max(1, round(dpi / 100))
        analysis = crop[::step, ::step]
        lum = gray(analysis)
        block = max(8, round(0.21 * dpi / step))
        _, dark_bg = background(lum, block)
        _, light_bg = background(255 - lum, block)
        # Pale antialiased red/blue edges can pass an absolute chroma threshold.
        # Recovery is stricter than gate triage: it must not turn coloured marks
        # into a new transcription. This is still not signature/seal detection.
        minimum = analysis.min(2).astype(float)
        chroma = analysis.max(2).astype(float) - minimum
        neutral = chroma <= np.maximum(6, 0.1 * (255 - minimum))
        dark = text_mask(dark_bg - lum >= 16, analysis, dpi / step) & neutral
        # White paper between coloured strokes is not white text. Require a dark
        # local median, as well as line structure, before inverting polarity.
        _, median = background(lum, block, percentile=50)
        dark_context = median < 160
        light = text_mask(light_bg - (255 - lum) >= 16, analysis, dpi / step) & neutral & dark_context
        if has_text_lines(dark) or has_text_lines(light):
            result.append({"picture_block": picture_index, "bbox": list(rect), "image": crop})
            covered_boxes.append(rect)  # Nested/overlapping pictures must not repeat pixels.
    return result
