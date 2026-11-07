"""Image helpers of the page router.

Ported from legal-vi-ocr `src/legal_vi_ocr/page_geometry.py` at 6e7ad59: only
`gray`, `background`, `_long_runs` and `text_mask`, which the quality gate and
picture recovery use. Orientation/warp estimation is not ported (ADR-0004).
"""

from __future__ import annotations

import numpy as np

DPI = 150
BLOCK = 32  # background tile size in pixels at DPI


def gray(rgb):
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.float32)


def background(lum, block=BLOCK, *, percentile=90):
    """Per-tile brightness (paper/p90 by default) and its full-resolution map."""
    h, w = lum.shape
    bh, bw = max(1, h // block), max(1, w // block)
    th, tw = h // bh, w // bw
    tiles = (
        lum[: bh * th, : bw * tw].reshape(bh, th, bw, tw).transpose(0, 2, 1, 3).reshape(bh, bw, -1)
    )
    grid = np.percentile(tiles, percentile, axis=2)
    full = np.repeat(np.repeat(grid, th, 0), tw, 1)
    full = np.pad(full, ((0, h - full.shape[0]), (0, w - full.shape[1])), mode="edge")
    return grid, full


def _long_runs(ink, length, axis):
    """Pixels inside straight runs of at least ``length`` ink pixels along ``axis`` (1: rows)."""
    a = ink if axis == 1 else ink.T
    if a.shape[1] < length:
        return np.zeros_like(ink)
    c = np.cumsum(np.pad(a, ((0, 0), (1, 0))), axis=1, dtype=np.int32)
    full = (c[:, length:] - c[:, :-length]) == length  # run window starting at each column
    s = np.cumsum(np.pad(full, ((0, 0), (1, 0))), axis=1, dtype=np.int32)
    j = np.arange(a.shape[1])
    lo, hi = np.clip(j - length + 1, 0, full.shape[1]), np.clip(j + 1, 0, full.shape[1])
    cover = (s[:, hi] - s[:, lo]) > 0
    return cover if axis == 1 else cover.T


def text_mask(ink, rgb, dpi=DPI):
    """Ink likely to be printed text: drop coloured marks (stamps, signatures) and ruling lines."""
    chroma = rgb.max(2).astype(np.int16) - rgb.min(2).astype(np.int16)
    length = max(8, round(0.27 * dpi))  # ~4.6 mm, longer than any glyph stroke at body sizes
    text = ink & (chroma < 60)
    return text & ~_long_runs(text, length, 1) & ~_long_runs(text, length, 0)
