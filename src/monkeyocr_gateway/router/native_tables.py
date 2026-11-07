"""Tables on pages with an embedded text layer: the PyMuPDF ``Page.find_tables()`` adapter.

Ported from legal-vi-ocr `src/legal_vi_ocr/native_tables.py` at 6e7ad59: the adapter, grid
post-processing (issue #34 there) and the token comparison it uses. The corpus survey, the
reviewed-page lists and the text check CLI are not ported.
"""

import unicodedata
from collections import Counter

import pymupdf

from .canonical import validate_page
from .insertions import _chars, cell_text, GAP_SPACE, moved_spans, needs_review, reorder
from .native import INSERTIONS, PyMuPDFAdapter

# find_tables() otherwise prints an advisory to stdout, mixing it into CLI output paths.
if hasattr(pymupdf, "no_recommend_layout"):
    pymupdf.no_recommend_layout()

TOLERANCE = 1.0


def _cells(table, raw=None):
    """Canonical cells with spans from a PyMuPDF table; covered positions are None in ``table.rows``.

    Cells holding a moved late-inserted span take their text from the reordered ``raw`` page instead
    of ``Table.extract()``, which interleaves the span with the whitespace it covers.
    """
    rows = table.rows
    n_rows, n_cols = len(rows), table.col_count
    col_x0 = [None] * n_cols
    for row in rows:
        for j, cell in enumerate(row.cells):
            if cell is not None and col_x0[j] is None:
                col_x0[j] = cell[0]
    if any(x is None for x in col_x0):
        raise ValueError("PyMuPDF table has a column without a starting cell")
    row_y0 = [row.bbox[1] for row in rows]
    texts = table.extract()
    moved = moved_spans(raw) if raw is not None else []
    cells, occupied = [], set()
    header = not table.header.external
    for r, row in enumerate(rows):
        for c, box in enumerate(row.cells):
            if box is None:
                continue
            colspan = sum(1 for j in range(c, n_cols) if col_x0[j] < box[2] - TOLERANCE)
            rowspan = sum(1 for i in range(r, n_rows) if row_y0[i] < box[3] - TOLERANCE)
            colspan, rowspan = max(colspan, 1), max(rowspan, 1)
            positions = {(i, j) for i in range(r, r + rowspan) for j in range(c, c + colspan)}
            if positions & occupied:
                raise ValueError("overlapping PyMuPDF table cells")
            occupied.update(positions)
            text = texts[r][c] or ""
            if raw is not None:
                native_text = cell_text(raw, box)
                # Table.extract() can join a separately embedded footnote digit
                # into a word (sample C). Use the original spans in that cell,
                # not linguistic repair or deletion of the small/hidden glyph.
                if any(_inside(s["bbox"], box) for s in moved) or Counter(tokens(text)) != Counter(
                    tokens(native_text)
                ):
                    text = native_text
            cells.append(
                {
                    "row": r,
                    "column": c,
                    "rowspan": rowspan,
                    "colspan": colspan,
                    "header": header and r == 0,
                    "text": text,
                }
            )
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) not in occupied:
                cells.append(
                    {"row": r, "column": c, "rowspan": 1, "colspan": 1, "header": False, "text": ""}
                )
    return {"rows": n_rows, "columns": n_cols, "cells": cells, "links": []}


def _inside(box, rect):
    x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return (
        rect[0] - TOLERANCE <= x <= rect[2] + TOLERANCE
        and rect[1] - TOLERANCE <= y <= rect[3] + TOLERANCE
    )


# Post-processing of find_tables() output (issue #34) ------------------------------------------------

POSTPROCESS = {
    "grid": "merge header rows whose only own cell has text into the cell above; drop rows/columns whose "
    "own single-span cells are all empty; shrink crossing spans",
    "validity": "tables with fewer than 2 rows, 2 columns or 2 non-empty cells, a cell boundary cutting a word, "
    "overlapping cells or native token recall below 0.99 fall back to text blocks; "
    "non-horizontal native spans skip table detection",
    "cell_text": "use original native cell spans when Table.extract changes token boundaries; never repair text",
    "continuation": "a table first on the page is flagged for review when the previous page ends with a table "
    "or its first row has an empty first cell; empty cells are never filled",
    "display": "views join cell lines with a space, or directly after a line-final hyphen; JSON keeps lines",
}
# Lines above a table that do not stop it from being the first content: page numbers and running headers.
FURNITURE_TOKENS = 3
FURNITURE_BAND = 0.1  # share of page height at the top and bottom


def _axis(name):
    return ("row", "rowspan", "rows") if name == "row" else ("column", "colspan", "columns")


def _covering(cells, axis, index):
    pos, span, _ = _axis(axis)
    return [c for c in cells if c[pos] <= index < c[pos] + c[span]]


def _removable(data, axis, index):
    """Remove empty grid axes, including slivers crossed by spanning cells. Never remove the only
    remaining row/column covered by a non-empty cell; crossing spans shrink without changing text."""
    _, span, count = _axis(axis)
    return data[count] > 1 and all(
        c[span] > 1 or not c["text"].strip() for c in _covering(data["cells"], axis, index)
    )


def _remove(data, axis, index):
    pos, span, count = _axis(axis)
    cells = []
    for c in data["cells"]:
        if c[pos] <= index < c[pos] + c[span]:
            if c[span] == 1:
                continue
            c = {**c, span: c[span] - 1}
        elif c[pos] > index:
            c = {**c, pos: c[pos] - 1}
        cells.append(c)
    return {**data, "cells": cells, count: data[count] - 1}


def _merge_header_row(data, index):
    """Merge row ``index`` into the row above when its only own cell holds text and every other position
    is covered by a cell starting in the first row (a header cell split by ruling lines into lines)."""
    cells = data["cells"]
    own = [c for c in cells if c["row"] == index]
    covered = [c for c in _covering(cells, "row", index) if c["row"] < index]
    if len(own) != 1 or not covered or any(c["row"] != 0 for c in covered):
        return None
    cell = own[0]
    if cell["rowspan"] != 1 or not cell["text"].strip():
        return None
    above = [
        c
        for c in cells
        if c["row"] == 0
        and c["rowspan"] == index
        and c["column"] == cell["column"]
        and c["colspan"] == cell["colspan"]
    ]
    if len(above) != 1 or not above[0]["header"]:
        return None
    top = above[0]
    text = "\n".join(t for t in (top["text"], cell["text"]) if t)
    merged = [
        {**c, "rowspan": c["rowspan"] + 1, "text": text} if c is top else c
        for c in cells
        if c is not cell
    ]
    return _remove({**data, "cells": merged}, "row", index)


def clean_grid(data):
    """Remove grid artefacts of ruling lines; text and its cell order are unchanged. Returns (data, notes)."""
    notes = Counter()
    changed = True
    while changed:
        changed = False
        for index in range(1, data["rows"]):
            merged = _merge_header_row(data, index)
            if merged is not None:
                data, changed = merged, True
                notes["merged_header_rows"] += 1
                break
        if changed:
            continue
        for axis in ("row", "column"):
            count = _axis(axis)[2]
            for index in range(data[count] - 1, -1, -1):
                if _removable(data, axis, index):
                    data, changed = _remove(data, axis, index), True
                    notes[f"removed_empty_{count}"] += 1
    data = {**data, "cells": sorted(data["cells"], key=lambda c: (c["row"], c["column"]))}
    return data, dict(notes)


def shape_problem(data):
    """Why a cleaned table is not a usable table, or None."""
    if sum(bool(c["text"].strip()) for c in data["cells"]) < 2:
        return "single_cell"
    if data["rows"] < 2:
        return "single_row"
    if data["columns"] < 2:
        return "single_column"
    return None


def cut_words(raw, table):
    """Words whose characters fall into different cells, i.e. a cell boundary cuts the word.

    Words are runs of characters split at whitespace or at a horizontal gap wider than a word gap, so text
    of adjacent cells written without a space is not taken for one word.
    """
    boxes = [c for c in table.cells if c is not None]
    area = pymupdf.Rect(table.bbox)

    def cell_of(box):
        x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return next(
            (k for k, b in enumerate(boxes) if b[0] <= x <= b[2] and b[1] <= y <= b[3]), None
        )

    found = []

    def flush(word):
        hits = {cell_of(c["bbox"]) for c in word} - {None}
        if len(hits) > 1:
            found.append("".join(c["c"] for c in word))

    for block in raw["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            if not area.intersects(line["bbox"]):
                continue
            word = []
            for span in line["spans"]:
                gap = GAP_SPACE * (span.get("size") or (span["bbox"][3] - span["bbox"][1]))
                for c in _chars(span):
                    if c["c"].isspace() or (word and c["bbox"][0] - word[-1]["bbox"][2] > gap):
                        flush(word)
                        word = []
                    if not c["c"].isspace():
                        word.append(c)
            flush(word)
    return found


def _line_text(line):
    return "".join(
        span["text"] if "text" in span else "".join(c["c"] for c in span["chars"])
        for span in line["spans"]
    )


def _content(line, height):
    """False for lines that look like page furniture: few tokens, or inside the top/bottom band."""
    box = line["bbox"]
    if len(_line_text(line).split()) <= FURNITURE_TOKENS:
        return False
    return not (box[3] <= FURNITURE_BAND * height or box[1] >= (1 - FURNITURE_BAND) * height)


def _lines(raw):
    return [l for b in raw["blocks"] if b["type"] == 0 for l in b["lines"]]


def first_on_page(raw, bbox):
    return not any(
        _content(l, raw["height"]) and l["bbox"][3] <= bbox[1] + TOLERANCE for l in _lines(raw)
    )


def ends_with_table(pdf_page):
    """True when the page's last content is a find_tables() table with at least 2 rows and 2 columns."""
    raw = pdf_page.get_text("dict")
    tables = [t for t in pdf_page.find_tables().tables if t.row_count >= 2 and t.col_count >= 2]
    if not tables:
        return False
    bottom = max(t.bbox[3] for t in tables)
    return not any(
        _content(l, raw["height"]) and l["bbox"][1] >= bottom - TOLERANCE for l in _lines(raw)
    )


def first_row_continues(data):
    """The first row, or the first row after the header, has an empty first cell but text elsewhere:
    a row continued from the previous page (sample C p3, p4). Its empty cells are left empty."""
    by_row = {}
    for c in data["cells"]:
        by_row.setdefault(c["row"], []).append(c)
    header = max((c["rowspan"] for c in by_row.get(0, [])), default=0)
    for row in {0, header}:
        own = sorted(by_row.get(row, []), key=lambda c: c["column"])
        if (
            own
            and own[0]["column"] == 0
            and not own[0]["text"].strip()
            and any(c["text"].strip() for c in own)
        ):
            return True
    return False


class PyMuPDFTablesAdapter(PyMuPDFAdapter):
    """PyMuPDF text blocks plus ``Page.find_tables()`` tables, default (ruling-line) settings."""

    def __init__(self):
        super().__init__()
        self.provenance = {
            "engine": "pymupdf",
            "version": pymupdf.VersionBind,
            "model": None,
            "revision": None,
            "config": {
                "text_mode": "rawdict",
                "insertions": INSERTIONS,
                "tables": "Page.find_tables()",
                "find_tables": {"strategy": "lines"},
                "header": "first row when PyMuPDF reports an internal header",
                "postprocess": POSTPROCESS,
            },
        }

    def extract(self, pdf_page, file, number):
        raw = reorder(pdf_page.get_text("rawdict"))[0]
        if any(
            abs(line.get("dir", (1, 0))[1]) > 0.01 or line.get("dir", (1, 0))[0] < 0.99
            for b in raw["blocks"]
            if b["type"] == 0
            for line in b["lines"]
        ):
            # Do not feed rotated spans to a horizontal-grid detector (#29).
            page = self.convert(raw, file, number)
            note = {
                "reason": "nonhorizontal_text",
                "review": True,
                "bbox": [0, 0, raw["width"], raw["height"]],
            }
            for block in page["blocks"]:
                block["provenance"] = self._with("table_fallback", note)
                block["quality"]["review"] = True
            page["quality"]["review"] = True
            return page
        tables = pdf_page.find_tables().tables
        previous = None
        if tables and pdf_page.number > 0 and any(first_on_page(raw, t.bbox) for t in tables):
            previous = ends_with_table(pdf_page.parent[pdf_page.number - 1])
        return self.convert_tables(raw, tables, file, number, previous)

    def table(self, raw, table, previous=None):
        """Canonical table data and post-processing notes, or (None, fallback reason) for text blocks."""
        try:
            data = _cells(table, raw)
        except ValueError as exc:
            return None, {"reason": "invalid_geometry", "error": str(exc), "review": True}
        original_tokens = Counter(
            t for row in table.extract() for text in row for t in tokens(text or "")
        )
        native_tokens = Counter(t for c in data["cells"] for t in tokens(c["text"]))
        data, notes = clean_grid(data)
        if original_tokens != native_tokens:
            notes["text_source"] = (
                "native cell spans; Table.extract() changed token boundaries or late-inserted span order"
            )
        problem = shape_problem(data)
        if problem:
            return None, {"reason": problem, "review": False}
        cut = cut_words(raw, table)
        if cut:
            return None, {"reason": "cell_cuts_word", "examples": cut[:10], "review": True}
        # Compare against the exact lines this candidate would suppress, not a
        # repaired OCR string. Counter recall catches omissions and word joins.
        source = "\n".join(
            "".join(s["text"] for s in line["spans"])
            for b in raw["blocks"]
            if b["type"] == 0
            for line in b["lines"]
            if _inside(line["bbox"], table.bbox)
        )
        result = compare(tokens("\n".join(c["text"] for c in data["cells"])), tokens(source))
        if result["recall"] is not None and result["recall"] < 0.99:
            return None, {
                "reason": "text_preservation",
                "review": True,
                "recall": result["recall"],
                "missing_examples": result["missing_examples"],
            }
        if first_on_page(raw, table.bbox):
            signals = {
                "previous_page_ends_with_table": previous,
                "first_row_continues": first_row_continues(data),
            }
            if previous or signals["first_row_continues"]:
                notes["continuation"] = {
                    "review": "may continue a table from the previous page; empty cells "
                    "are not filled",
                    **signals,
                }
        return data, notes

    def _with(self, key, value):
        return {**self.provenance, "config": {**self.provenance["config"], key: value}}

    def convert_tables(self, raw, tables, file, number, previous=None):
        """``previous``: whether the previous page ends with a table (None when unknown or not needed)."""
        quality = {"confidence": None, "truncated": False, "review": False}
        results = []
        for t in tables:
            data, notes = self.table(raw, t, previous)
            results.append(
                (
                    list(t.bbox),
                    data,
                    {**notes, "bbox": [round(v, 2) for v in t.bbox]} if data is None else notes,
                )
            )
        emitted, blocks = set(), []

        def table_block(index):
            bbox, data, notes = results[index]
            text = "\n".join(c["text"] for c in data["cells"] if c["text"])
            emitted.add(index)
            provenance = self._with("table_postprocess", notes) if notes else self.provenance
            return {
                "type": "table",
                "bbox": bbox,
                "coordinate_unit": "pt",
                "raw_text": text,
                "table": data,
                "provenance": provenance,
                "quality": {**quality, "review": True},
            }

        def text_block(lines, fallback):
            block = self._text(lines, quality)
            if fallback is not None:
                notes = results[fallback][2]
                block["provenance"] = self._with("table_fallback", notes)
                block["quality"]["review"] |= notes["review"]
            return block

        for block in raw["blocks"]:
            if block["type"] != 0:
                continue
            group, key = (
                [],
                None,
            )  # consecutive lines outside tables, or inside the same rejected table
            for line in block["lines"]:
                hit = next(
                    (i for i, (bbox, _, _) in enumerate(results) if _inside(line["bbox"], bbox)),
                    None,
                )
                if hit is not None and results[hit][1] is not None:
                    if hit not in emitted:
                        if group:
                            blocks.append(text_block(group, key))
                            group, key = [], None
                        blocks.append(table_block(hit))
                    continue
                if group and hit != key:
                    blocks.append(text_block(group, key))
                    group = []
                group.append(line)
                key = hit
            if group:
                blocks.append(text_block(group, key))
        for index in range(len(results)):
            if index not in emitted and results[index][1] is not None:
                block = table_block(index)  # a table without text lines is placed by its top edge
                at = next(
                    (k for k, b in enumerate(blocks) if b["bbox"][1] > block["bbox"][1]),
                    len(blocks),
                )
                blocks.insert(at, block)
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

    def _text(self, lines, quality):
        box = [
            min(l["bbox"][0] for l in lines),
            min(l["bbox"][1] for l in lines),
            max(l["bbox"][2] for l in lines),
            max(l["bbox"][3] for l in lines),
        ]
        text = "\n".join("".join(span["text"] for span in line["spans"]) for line in lines)
        return {
            "type": "text",
            "bbox": box,
            "coordinate_unit": "pt",
            "raw_text": text,
            "provenance": self.provenance,
            "quality": {**quality, "review": needs_review(lines)},
        }


# Token comparison ----------------------------------------------------------------------------------


def tokens(text):
    return unicodedata.normalize("NFC", text).split()


def compare(output_tokens, native_tokens):
    out, nat = Counter(output_tokens), Counter(native_tokens)
    matched = sum((out & nat).values())
    native_set = set(nat)
    unmatched = [t for t in output_tokens if t not in native_set]
    # Joined words: an output token equal to 2-3 consecutive native words, e.g. "phòngUBND".
    joins = set()
    for size in (2, 3):
        for i in range(len(native_tokens) - size + 1):
            joins.add("".join(native_tokens[i : i + size]))
    joined = [t for t in unmatched if t in joins]
    # Split words: consecutive unmatched output tokens whose concatenation is a native word.
    split = []
    for size in (2, 3):
        for i in range(len(output_tokens) - size + 1):
            window = output_tokens[i : i + size]
            if any(t not in native_set for t in window) and "".join(window) in native_set:
                split.append(" ".join(window))
    return {
        "output_tokens": len(output_tokens),
        "native_tokens": len(native_tokens),
        "matched": matched,
        "precision": matched / len(output_tokens) if output_tokens else None,
        "recall": matched / len(native_tokens) if native_tokens else None,
        "unmatched_tokens": len(unmatched),
        "joined_words": joined,
        "split_words": split,
        "unmatched_examples": unmatched[:20],
        "missing_examples": list((nat - out).elements())[:20],
    }
