"""Per-page text-layer profile used by the quality gate.

Ported from legal-vi-ocr `src/legal_vi_ocr/profiler.py` at 6e7ad59: only
`profile_page` and its character sets. Corpus sampling is not ported.
"""

from __future__ import annotations

import unicodedata

VIETNAMESE = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
MARKS = set("\u0300\u0301\u0303\u0309\u0323\u0302\u0306\u031b")


def profile_page(page, name, number):
    raw = page.get_text()
    normalized = unicodedata.normalize("NFC", raw)
    valid = sum(ch.lower() in VIETNAMESE for ch in normalized)
    invalid = sum(ch in MARKS for ch in normalized)
    count = sum(not ch.isspace() for ch in raw)
    images = [list(map(lambda n: round(n, 2), image["bbox"])) for image in page.get_image_info()]
    return {
        "file": name,
        "page": number,
        "page_type": "native" if count else "scan",
        "character_count": count,
        "valid_vietnamese_diacritic_ratio": valid / (valid + invalid) if valid + invalid else None,
        "invalid_diacritic_marks": invalid,
        "bad_glyph_count": sum(
            ch == "\ufffd" or (unicodedata.category(ch) == "Cc" and not ch.isspace()) for ch in raw
        ),
        "image_regions": images,
    }
