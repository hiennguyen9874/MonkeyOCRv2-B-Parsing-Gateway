"""Synthetic PDFs and images for the page router tests (no corpus material).

Builders ported from legal-vi-ocr tests at 6e7ad59 (`test_router.py`,
`test_router_hardening.py`, `test_native_tables.py`, `test_insertions.py`,
`test_picture_recovery.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pymupdf

DIAGRAM = json.loads((Path(__file__).parent / "fixtures/router-diagram.json").read_text())


def text_image_png(width=240, height=130, lines=("Missing image text line number one",
                                                 "Missing image text line number two"),
                   fontsize=11, color=(0, 0, 0), dpi=150, origin=(20, 50), step=40) -> bytes:
    with pymupdf.open() as doc:
        page = doc.new_page(width=width, height=height)
        for i, line in enumerate(lines):
            page.insert_text((origin[0], origin[1] + step * i), line, fontsize=fontsize, color=color)
        return page.get_pixmap(dpi=dpi, alpha=False).tobytes("png")


def table_pdf(doc=None):
    """One page: text, a ruled 2x3 table whose top-right cell spans two columns, text."""
    doc = doc or pymupdf.open()
    page = doc.new_page(width=400, height=300)
    page.insert_text((50, 30), "Heading above", fontsize=10)
    for y in (50, 100, 150):
        page.draw_line((50, y), (350, y))
    for x in (50, 150, 350):
        page.draw_line((x, 50), (x, 150))
    page.draw_line((250, 100), (250, 150))
    page.insert_text((55, 75), "Merged", fontsize=10)
    page.insert_text((155, 75), "Wide header", fontsize=10)
    for x, text in ((55, "a1"), (155, "b1"), (255, "c1")):
        page.insert_text((x, 125), text, fontsize=10)
    page.insert_text((50, 200), "Text below", fontsize=10)
    return doc


def insertion_page(doc):
    """A header and a table cell with blanks, filled by spans drawn last (as in stamped PDFs)."""
    page = doc.new_page(width=400, height=300)
    header = "Kem theo Ke hoach so          /KH-SKHCN ngay         /10/2025"
    page.insert_text((50, 30), header, fontsize=10)
    for y in (50, 100, 150):
        page.draw_line((40, y), (380, y))
    for x in (40, 200, 380):
        page.draw_line((x, 50), (x, 150))
    page.insert_text((45, 75), "So          /QD ngay", fontsize=10)
    page.insert_text((205, 75), "Dong", fontsize=10)
    page.insert_text((45, 125), "a1", fontsize=10)
    page.insert_text((205, 125), "b1", fontsize=10)
    page.insert_text((50, 200), "Text below", fontsize=10)
    page.insert_text((50, 250), "Overprinted line", fontsize=10)
    width = lambda text: pymupdf.get_text_length(text, fontsize=10)  # noqa: E731
    page.insert_text((50 + width("Kem theo Ke hoach so ") + 2, 29.5), "2186", fontsize=10)
    page.insert_text((50 + width("Kem theo Ke hoach so          /KH-SKHCN ngay ") + 4, 29), "29", fontsize=10)
    page.insert_text((45 + width("So ") + 2, 75), "15", fontsize=10)
    page.insert_text((50 + width("Over"), 250), "XX", fontsize=10)  # covers visible text: not placed
    return page


def diagram(dpi=144):
    """All labels are synthetic; include white-on-colour and dark-on-white text."""
    with pymupdf.open() as doc:
        p = doc.new_page(width=300, height=160)
        p.draw_rect((0, 0, 300, 160), fill=(0.15, 0.3, 0.45))
        p.insert_text((20, 30), DIAGRAM["heading"], color=(1, 1, 1), fontsize=14)
        for i, text in enumerate(DIAGRAM["labels"]):
            y = 40 + i * 36
            p.draw_rect((15, y, 285, y + 30), fill=(1, 1, 1))
            p.insert_text((30, y + 21), text, fontsize=12)
        pix = p.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
