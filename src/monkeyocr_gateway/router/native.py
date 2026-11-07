"""Native PDF text adapter.

Ported verbatim from legal-vi-ocr `src/legal_vi_ocr/native.py` at 6e7ad59.
"""

from abc import ABC, abstractmethod

import pymupdf

from .canonical import validate_page
from .insertions import needs_review, reorder

# Late-inserted spans (document number, date) are moved onto the line they overlay; see insertions.py.
INSERTIONS = "late-inserted spans moved onto the line they overlay when unambiguous; others flagged for review (issue #33)"


class EngineAdapter(ABC):
    @abstractmethod
    def extract(self, pdf_page, file, number):
        """Extract a page into the canonical contract."""


class PyMuPDFAdapter(EngineAdapter):
    def __init__(self):
        self.provenance = {
            "engine": "pymupdf",
            "version": pymupdf.VersionBind,
            "model": None,
            "revision": None,
            "config": {"text_mode": "rawdict", "insertions": INSERTIONS},
        }

    def extract(self, pdf_page, file, number):
        return self.convert(reorder(pdf_page.get_text("rawdict"))[0], file, number)

    def convert(self, raw, file, number):
        quality = {"confidence": None, "truncated": False, "review": False}
        blocks = []
        for block in raw["blocks"]:
            if block["type"] != 0:
                continue
            text = "\n".join(
                "".join(span["text"] for span in line["spans"]) for line in block["lines"]
            )
            blocks.append(
                {
                    "type": "text",
                    "bbox": list(block["bbox"]),
                    "coordinate_unit": "pt",
                    "raw_text": text,
                    "provenance": self.provenance,
                    "quality": {**quality, "review": needs_review(block["lines"])},
                }
            )
        page = {
            "schema_version": "1.0",
            "file": file,
            "documents": [],
            "page": {
                "number": number,
                "width": raw["width"],
                "height": raw["height"],
                "coordinate_unit": "pt",
                "type": None,
                "routing": None,
            },
            "blocks": blocks,
            "provenance": self.provenance,
            "quality": {**quality, "review": any(b["quality"]["review"] for b in blocks)},
        }
        validate_page(page)
        return page
