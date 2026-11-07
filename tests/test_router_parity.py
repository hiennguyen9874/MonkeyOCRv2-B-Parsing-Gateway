"""Parity tests of the page router helpers ported from legal-vi-ocr at 6e7ad59 (ADR-0007).

Each test reuses a synthetic test of the source repo (named in its docstring), so a
behaviour change in a ported helper shows up here. Tests of the batch runner's
manifest/replay/artifact paths are not ported: the gateway has no such paths.
"""

from __future__ import annotations

import asyncio
import copy
from unittest.mock import AsyncMock, patch

import numpy as np
import pymupdf
import pytest

from monkeyocr_gateway.router.canonical import validate_page
from monkeyocr_gateway.router.gate import assess, merged_regions
from monkeyocr_gateway.router.insertions import reorder
from monkeyocr_gateway.router.native import PyMuPDFAdapter
from monkeyocr_gateway.router.native_tables import (
    PyMuPDFTablesAdapter,
    _cells,
    clean_grid,
    compare,
    ends_with_table,
    first_row_continues,
    shape_problem,
)
from monkeyocr_gateway.router.picture_recovery import candidates
from monkeyocr_gateway.router.review import annotate, spatial_merge
from monkeyocr_gateway.router.views import cell_display, table_html
from monkeyocr_gateway.routed import native_block, parse_ocr_request
from monkeyocr_gateway.vllm_client import GenerationOptions
from router_pdfs import diagram, insertion_page, table_pdf

FILE = {"path": "t.pdf", "sha256": "a" * 64}


# Quality gate: test_router.GateTest -------------------------------------------------------------

@pytest.mark.parametrize(
    "mode", ["visible", "hidden", "transparent", "white", "overlap", "occluded", "glyphs", "marks"]
)
def test_corrupt_embedded_layers_do_not_pass_on_character_count_alone(mode):
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=300)
        text = "A visible legal heading with enough characters"
        if mode == "marks":
            # A ToUnicode map exposing unattached combining accents.
            text = "A" * 45
        page.insert_text((20, 40), text, render_mode=3 if mode == "hidden" else 0,
                         fill_opacity=0 if mode == "transparent" else 1,
                         color=(1, 1, 1) if mode == "white" else (0, 0, 0))
        if mode == "overlap":
            page.insert_text((20, 40), text)
        if mode == "occluded":
            page.draw_rect(page.rect, fill=(1, 1, 1), overlay=True)
        if mode in ("glyphs", "marks"):
            xref = doc.get_new_xref()
            doc.update_object(xref, "<<>>")
            target = "E000" if mode == "glyphs" else "0301"
            codes = "\n".join(f"<{code:02X}> <{target}>" for code in range(33, 127))
            cmap = ("/CIDInit /ProcSet findresource begin 12 dict begin begincmap "
                    "/CMapType 2 def 1 begincodespacerange <00> <FF> endcodespacerange "
                    f"94 beginbfchar {codes} endbfchar endcmap end end")
            doc.update_stream(xref, cmap.encode())
            doc.xref_set_key(page.get_fonts()[0][0], "ToUnicode", f"{xref} 0 R")
            page = doc.reload_page(page)
        route = assess(page)
        assert route["signals"]["character_count"] > 20
        assert route["decision"] == ("native" if mode == "visible" else "ocr")
        if mode != "visible":
            expected = {"hidden": "invisible_text", "transparent": "invisible_text", "white": "invisible_text",
                        "overlap": "overlapping_text", "occluded": "invisible_text",
                        "glyphs": "bad_glyphs", "marks": "invalid_vietnamese_diacritics"}[mode]
            assert expected in route["reasons"]


def test_ascii_is_not_required_to_have_accents():
    with pymupdf.open() as doc:
        p = doc.new_page()
        p.insert_text((20, 40), "12345 Legal ASCII heading with enough text")
        r = assess(p)
        assert r["decision"] == "native"
        assert r["signals"]["valid_vietnamese_diacritic_ratio"] is None


def test_merged_placements_are_transitive_and_prioritized_by_area():
    # test_router_hardening.test_merged_placements_are_transitive_and_prioritized_by_area
    regions = merged_regions([[0, 0, 10, 10], [20, 0, 30, 10], [9, 0, 21, 10], [40, 0, 41, 1]])
    assert [list(r) for r in regions] == [[0, 0, 30, 10], [40, 0, 41, 1]]


# Review and ordering: test_router_hardening ----------------------------------------------------

def _block(text, bbox, ocr=False, label="Text"):
    return {"bbox": bbox, "label": label, "content": text, "raw_text": text, "source": "ocr" if ocr else "native",
            "review": False, "config": {"routed_request": "x"} if ocr else {}}


def test_spatial_anchors_preserve_native_sequence_and_sort_only_supplements():
    # test_router_hardening.test_spatial_anchors_preserve_native_sequence_and_sort_only_supplements
    # Native right-column header deliberately comes before left-column header.
    native = [_block("right", [300, 0, 500, 50]), _block("left", [0, 0, 200, 50]),
              _block("footer", [0, 700, 500, 750])]
    page = {"blocks": native + [_block("lower", [10, 300, 200, 350], True),
                                _block("upper", [10, 100, 200, 150], True)]}
    spatial_merge(page)
    assert [b["raw_text"] for b in page["blocks"]] == ["right", "left", "upper", "lower", "footer"]
    assert [b for b in page["blocks"] if not b["config"]] == native


def test_full_page_ocr_keeps_model_reading_order():
    """Gateway difference: without native Khối, OCR keeps Monkey's reading order (two columns)."""
    blocks = [_block("left top", [0, 0, 200, 50], True), _block("left bottom", [0, 400, 200, 450], True),
              _block("right top", [300, 0, 500, 50], True)]
    page = {"blocks": list(blocks)}
    spatial_merge(page)
    assert page["blocks"] == blocks


@pytest.mark.parametrize("labels,issues", [
    ([], ["empty_output"]),
    (["Text"], []),
    (["Text", "Picture"], ["picture_review"]),
    (["Text", "Table"], ["table_review"]),
])
def test_ocr_quality_matches_source_convert_records(labels, issues):
    """legal-vi-ocr monkey_two_stage.convert_records / monkey.convert at 6e7ad59.

    Empty output flags the page; a table/picture flags every block, not just itself.
    The gateway carries page quality as request issues without altering predictions.
    """
    records = [{"label": label, "bbox": [0, 0, 100, 100], "content": "raw prediction"}
               for label in labels]
    before = copy.deepcopy(records)
    request = {"masked_native_words": False}
    with patch("monkeyocr_gateway.routed.parse_page", new_callable=AsyncMock, return_value=records):
        blocks = asyncio.run(parse_ocr_request(None, request, None, GenerationOptions(), 0))
    assert request["issues"] == issues
    assert [b["review"] for b in blocks] == [bool(issues)] * len(records)
    assert records == before
    assert [{key: block[key] for key in ("label", "bbox", "content")} for block in blocks] == records


def _routed_native_page(p):
    canonical = PyMuPDFAdapter().extract(p, FILE, 1)
    return {"blocks": [native_block(b) for b in canonical["blocks"]], "routing": assess(p),
            "review": canonical["quality"]["review"]}


def test_review_anomalies_keep_raw_text_and_table_cells():
    # test_router_hardening.test_review_anomalies_keep_raw_text_and_table_cells
    with pymupdf.open() as doc:
        p = doc.new_page()
        p.insert_text((20, 50), "Native legal heading with enough text")
        page = _routed_native_page(p)
    block = page["blocks"][0]
    block["raw_text"] = block["content"] = "QUYẾT ĐỊNH СỘNG HÒА α"  # Cyrillic lookalikes, Greek
    before = copy.deepcopy(block)
    annotate(page)
    assert page["blocks"][0] == before
    codes = [r["code"] for r in page["routing"]["review_reasons"]]
    assert "unexpected_script" in codes
    assert "mixed_script_word" in codes
    assert page["review"]
    # A script anomaly present only in a table cell is still reviewed.
    table = {"rows": 1, "columns": 1, "links": [],
             "cells": [{"row": 0, "column": 0, "rowspan": 1, "colspan": 1, "header": False, "text": "СỘNG"}]}
    block.update(native_block({"type": "table", "bbox": block["bbox"], "raw_text": "", "table": table,
                               "provenance": {"config": {}}, "quality": {"review": False}}))
    page["routing"]["review_reasons"] = []
    before = copy.deepcopy(block)
    annotate(page)
    assert block == before
    assert "unexpected_script" in [r["code"] for r in page["routing"]["review_reasons"]]
    assert "table_requires_review" in [r["code"] for r in page["routing"]["review_reasons"]]


# Late-inserted spans: test_insertions.AdaptersTest -------------------------------------------

@pytest.mark.parametrize("adapter", [PyMuPDFAdapter(), PyMuPDFTablesAdapter()], ids=["text", "tables"])
def test_text_blocks_table_cells_and_review(adapter):
    with pymupdf.open() as doc:
        page = adapter.extract(insertion_page(doc), FILE, 1)
    validate_page(page)
    texts = [b["raw_text"] for b in page["blocks"]]
    assert texts[0] == "Kem theo Ke hoach so 2186 /KH-SKHCN ngay 29 /10/2025"
    assert any(t.startswith("So 15 /QD ngay") for t in texts), texts
    over = next(b for b in page["blocks"] if b["raw_text"] == "XX")
    assert over["quality"]["review"]
    assert page["quality"]["review"]
    if isinstance(adapter, PyMuPDFTablesAdapter):
        table = next(b for b in page["blocks"] if b["type"] == "table")
        cell = next(c for c in table["table"]["cells"] if (c["row"], c["column"]) == (0, 0))
        assert cell["text"] == "So 15 /QD ngay"
        assert table["raw_text"] == "So 15 /QD ngay\nDong\na1\nb1"


def test_report_and_unchanged_characters():
    # test_insertions.RecordedPagesTest.test_report_and_unchanged_characters, on the synthetic page
    with pymupdf.open() as doc:
        raw = insertion_page(doc).get_text("rawdict")
    moved, report = reorder(raw)
    assert [m["text"] for m in report["moved"]] == ["2186", "29", "15"]
    assert [(u["text"], u["reason"]) for u in report["unplaced"]] == [("XX", "overlaps_text")]
    before = sorted(c["c"] for b in raw["blocks"] if b["type"] == 0 for l in b["lines"] for s in l["spans"]
                    for c in s["chars"] if not c["c"].isspace())
    after = sorted(ch for b in moved["blocks"] if b["type"] == 0 for l in b["lines"] for s in l["spans"]
                   for ch in s["text"] if not ch.isspace())
    assert before == after


# Native tables: test_native_tables and test_router_hardening ------------------------------------

def test_spans_and_reading_order():
    with table_pdf() as doc:
        page = PyMuPDFTablesAdapter().extract(doc[0], FILE, 1)
    validate_page(page)
    assert [b["type"] for b in page["blocks"]] == ["text", "table", "text"]
    table = page["blocks"][1]["table"]
    assert (table["rows"], table["columns"]) == (2, 3)
    wide = next(c for c in table["cells"] if c["text"] == "Wide header")
    assert (wide["column"], wide["colspan"], wide["header"]) == (1, 2, True)
    assert page["blocks"][1]["quality"]["review"]
    html = table_html(table)
    assert html.startswith("<table><tr><th")
    assert '<th rowspan="1" colspan="2">Wide header</th>' in html


def test_rejected_table_does_not_remove_valid_table_or_duplicate_text():
    with table_pdf() as doc:
        pdf_page = doc[0]
        for y in (180, 230):
            pdf_page.draw_line((50, y), (350, y))
        for x in (50, 200, 350):
            pdf_page.draw_line((x, 180), (x, 230))
        pdf_page.insert_text((205, 200), "Other", fontsize=10)
        page = PyMuPDFTablesAdapter().extract(pdf_page, FILE, 1)
    assert [b["type"] for b in page["blocks"]] == ["text", "table", "text"]
    assert page["blocks"][-1]["raw_text"] == "Text below\nOther"
    assert page["blocks"][-1]["provenance"]["config"]["table_fallback"]["reason"] == "single_row"


def test_compare_flags_joined_and_split_words():
    native = "Văn phòng UBND tỉnh nghị quyết".split()
    result = compare("Văn phòngUBND tỉnh ngh ị quyết".split(), native)
    assert result["joined_words"] == ["phòngUBND"]
    assert result["split_words"] == ["ngh ị"]
    assert result["precision"] < 1
    assert result["recall"] < 1


def test_native_cell_spans_keep_separate_digits_without_losing_real_table():
    with table_pdf() as doc:
        p = doc[0]
        p.insert_text((60, 90), "1", fontsize=1, render_mode=3)
        raw = reorder(p.get_text("rawdict"))[0]
        table = p.find_tables().tables[0]
        texts = table.extract()
        texts[0][0] = "Merged1"  # emulate Table.extract joining a separate glyph
        with patch.object(type(table), "extract", return_value=texts):
            page = PyMuPDFTablesAdapter().convert_tables(raw, [table], FILE, 1)
    result = next(b for b in page["blocks"] if b["type"] == "table")
    assert result["table"]["cells"][0]["text"].split() == ["Merged", "1"]
    assert "text_source" in result["provenance"]["config"]["table_postprocess"]
    assert "Merged1" not in result["raw_text"]


def test_table_text_preservation_failure_retains_native_lines():
    with table_pdf() as doc:
        p = doc[0]
        raw = reorder(p.get_text("rawdict"))[0]
        tables = p.find_tables().tables
        data = _cells(tables[0], raw)
        data["cells"][0]["text"] = "dropped original heading"
        with patch("monkeyocr_gateway.router.native_tables._cells", return_value=data):
            canonical = PyMuPDFTablesAdapter().convert_tables(raw, tables, FILE, 1)
        assert not any(b["type"] == "table" for b in canonical["blocks"])
        assert "Merged" in " ".join(b["raw_text"] for b in canonical["blocks"])
        page = {"blocks": [native_block(b) for b in canonical["blocks"]], "routing": assess(p),
                "review": canonical["quality"]["review"]}
    annotate(page)
    assert page["routing"]["table_fallbacks"][0]["reason"] == "text_preservation"
    assert "native_table_fallback" in [r["code"] for r in page["routing"]["review_reasons"]]
    assert page["review"]


def test_nonhorizontal_native_spans_skip_table_detector():
    with pymupdf.open() as doc:
        p = doc.new_page()
        p.insert_text((50, 500), "Rotated legal native text with enough characters", rotate=90)
        with patch.object(pymupdf.Page, "find_tables", side_effect=AssertionError("must not run")):
            page = PyMuPDFTablesAdapter().extract(p, FILE, 1)
    assert page["quality"]["review"]
    assert page["blocks"][0]["provenance"]["config"]["table_fallback"]["reason"] == "nonhorizontal_text"


def _cell(row, column, text="", rowspan=1, colspan=1):
    return {"row": row, "column": column, "rowspan": rowspan, "colspan": colspan, "header": False, "text": text}


def _grid(table):
    cells = {(c["row"], c["column"]): c for c in table["cells"]}
    return [[cells.get((r, c)) for c in range(table["columns"])] for r in range(table["rows"])]


def test_display_joins_lines():
    assert cell_display("Số\n1281/UBND-\nVHXH ngày") == "Số 1281/UBND-VHXH ngày"
    assert cell_display("- a\n- b") == "- a - b"
    assert cell_display("2026 -\n2030") == "2026 -2030"
    assert cell_display("") == ""


def test_degenerate_shapes_are_rejected():
    one_row = [_cell(0, c, "a") for c in range(3)]
    assert shape_problem({"rows": 1, "columns": 3, "cells": one_row}) == "single_row"
    one_col = [_cell(r, 0, "a") for r in range(3)]
    assert shape_problem({"rows": 3, "columns": 1, "cells": one_col}) == "single_column"
    one_text = [_cell(0, 0, "a"), _cell(0, 1), _cell(1, 0), _cell(1, 1)]
    assert shape_problem({"rows": 2, "columns": 2, "cells": one_text}) == "single_cell"
    two_cols = one_col + [_cell(r, 1, "b") for r in range(3)]
    assert shape_problem({"rows": 3, "columns": 2, "cells": two_cols}) is None


def test_empty_axes_removed_but_individual_empty_cells_preserved():
    cells = [_cell(r, c, "x" if (r, c) in {(1, 1), (2, 2)} else "") for r in range(3) for c in range(3)]
    data, notes = clean_grid({"rows": 3, "columns": 3, "cells": cells, "links": []})
    assert (data["rows"], data["columns"]) == (2, 2)
    assert notes == {"removed_empty_rows": 1, "removed_empty_columns": 1}
    assert [[c["text"] for c in row] for row in _grid(data)] == [["x", ""], ["", "x"]]


def test_data_row_with_one_cell_is_not_merged():
    cells = [_cell(0, 0, "A", rowspan=2), _cell(0, 1, "B"), _cell(1, 1, "c"), _cell(2, 0, "1"), _cell(2, 1, "d")]
    data, _ = clean_grid({"rows": 3, "columns": 2, "cells": cells, "links": []})
    assert data["rows"] == 3  # no header evidence: do not merge data
    for c in cells[:2]:
        c["header"] = True
    data, _ = clean_grid({"rows": 3, "columns": 2, "cells": cells, "links": []})
    assert data["rows"] == 2  # header "B" + "c" merged
    cells = [_cell(0, 0, "1"), _cell(0, 1, "a"), _cell(1, 0, "2", rowspan=2), _cell(1, 1, "b"), _cell(2, 1, "c")]
    data, _ = clean_grid({"rows": 3, "columns": 2, "cells": cells, "links": []})
    assert data["rows"] == 3  # the spanning cell does not start in the first row


def test_first_row_continues():
    cells = [_cell(0, 0, "STT"), _cell(0, 1, "Mô tả"), _cell(1, 0), _cell(1, 1, "tiếp")]
    assert first_row_continues({"rows": 2, "columns": 2, "cells": cells, "links": []})
    cells[2] = _cell(1, 0, "1")
    assert not first_row_continues({"rows": 2, "columns": 2, "cells": cells, "links": []})


def test_previous_page_ending_with_table():
    with pymupdf.open() as doc:
        for _ in range(2):
            page = doc.new_page(width=400, height=300)
            page.insert_text((50, 30), "Some running text above the table here", fontsize=10)
            for y in (120, 160, 200):
                page.draw_line((50, y), (350, y))
            for x in (50, 200, 350):
                page.draw_line((x, 120), (x, 200))
            for x in (55, 205):
                page.insert_text((x, 140), "cell", fontsize=10)
                page.insert_text((x, 180), "cell", fontsize=10)
            page.insert_text((190, 290), "- 1 -", fontsize=8)  # page number: furniture
        doc[1].insert_text((50, 240), "A closing paragraph below the table", fontsize=10)
        assert ends_with_table(doc[0])
        assert not ends_with_table(doc[1])


# Picture recovery: test_picture_recovery ----------------------------------------------------------

def test_both_polarities_masking_and_nontext_picture():
    rgb = diagram()
    box = [0, 0, rgb.shape[1], rgb.shape[0]]
    assert len(candidates(rgb, [(0, box)], [], 144)) == 1
    assert candidates(rgb, [(0, box)], [box], 144) == []
    # White heading on dark paper still needs recognition, even without body boxes.
    white_only = np.full_like(rgb, [38, 76, 115])
    white_only[:75] = rgb[:75]
    assert len(candidates(white_only, [(0, box)], [], 144)) == 1
    # A small Picture remains reviewed; this fallback must not become a new
    # stamp/signature reader. Primary all-size supplemental OCR is unchanged.
    assert candidates(rgb, [(0, box)], [], 300) == []
    solid = np.full_like(rgb, 80)
    assert candidates(solid, [(0, box)], [], 144) == []
    assert candidates(rgb, [], [], 144) == []
    assert len(candidates(rgb, [(0, box), (1, box)], [], 144)) == 1


def test_coloured_ring_and_pen_are_not_new_text_candidates():
    # Non-sensitive stand-in for the exploratory run's stamp hallucination.
    with pymupdf.open() as doc:
        p = doc.new_page(width=300, height=200)
        p.draw_circle((70, 70), 60, color=(0.9, 0.2, 0.2), width=3)
        p.draw_circle((70, 70), 52, color=(0.9, 0.2, 0.2), width=2)
        p.insert_text((25, 65), "SAMPLE SEAL", color=(0.9, 0.2, 0.2), fontsize=9)
        p.draw_polyline([(105, 100), (145, 35), (130, 90), (180, 50), (170, 75), (220, 30)],
                        color=(0, 0, 0.8), width=2)
        p.draw_line((100, 110), (290, 55), color=(0, 0, 0.8), width=2)
        pix = p.get_pixmap(dpi=144, colorspace=pymupdf.csRGB, alpha=False)
        rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
        assert candidates(rgb, [(0, [0, 0, pix.width, pix.height])], [], 144) == []
