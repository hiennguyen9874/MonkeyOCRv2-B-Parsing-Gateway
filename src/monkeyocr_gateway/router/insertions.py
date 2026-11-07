"""Late-inserted spans on native pages: put them back on the line they overlay (issue #33).

Some PDFs get the document number and date stamped later, over the blank in ``Kế hoạch số ___ ngày ___``.
In the content stream these spans come last, so PyMuPDF emits them at the end of the page. ``reorder``
moves such a span into the line it overlays when the geometry is unambiguous: it overlaps the line
vertically, lies between the line's first and last visible character, and covers only whitespace or
empty space of that line (a neighbouring glyph may touch its edge). Characters are never changed; the
whitespace run on each side of the span is collapsed to one space, so the span stays a separate word.
Spans that overlay a line but cannot be placed (they overlap visible characters or fit several lines)
are left where they are and flagged for review instead of guessed. Invisible text (alpha 0, e.g. an OCR
layer over a scan) is ignored.

Ported from legal-vi-ocr `src/legal_vi_ocr/insertions.py` at 6e7ad59, without the corpus survey CLI.
"""

import copy

import pymupdf

TOLERANCE = 1.0  # pt a span may extend past the first or last visible character of its host line
MIN_VERTICAL_OVERLAP = 0.5  # share of the shorter height
GAP_SPACE = 0.25  # empty space wider than this share of the font size is read as a word gap


def _space(c):
    return c.isspace()


def _visible(span):
    """Invisible text (alpha 0, e.g. an OCR layer over a scan) neither overlays nor hosts anything."""
    return span.get("alpha", 255) > 0


def _chars(span):
    """Character boxes of a span; ``dict`` output has none, so they are spread evenly over the bbox."""
    if "chars" in span:
        return span["chars"]
    text, (x0, y0, x1, y1) = span["text"], span["bbox"]
    width = (x1 - x0) / len(text) if text else 0
    return [{"c": c, "bbox": (x0 + i * width, y0, x0 + (i + 1) * width, y1)} for i, c in enumerate(text)]


def _text(span):
    return span["text"] if "text" in span else "".join(c["c"] for c in span["chars"])


def _union(boxes):
    boxes = list(boxes)
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _vertical_overlap(a, b):
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    shorter = min(a[3] - a[1], b[3] - b[1])
    return shorter > 0 and overlap >= MIN_VERTICAL_OVERLAP * shorter


def _relation(span, line, ink):
    """None if ``span`` does not overlay ``line``; otherwise "placeable" or a reason it cannot be placed."""
    box = span["bbox"]
    if not _vertical_overlap(box, line["bbox"]):
        return None
    if not ink:
        return None
    if not (min(c["bbox"][0] for c in ink) - TOLERANCE <= box[0]
            and box[2] <= max(c["bbox"][2] for c in ink) + TOLERANCE):
        return None
    # Glyph boxes include side bearings and stamped numbers are not aligned to the blank, so a host glyph
    # may stick into the span from the outside by a little (e.g. the "/" after "6240"). Any other overlap
    # with a visible host glyph (one covered, or deeper than a word gap) is a collision.
    allowance = GAP_SPACE * (span.get("size") or (box[3] - box[1]))
    for c in ink:
        x0, x1 = c["bbox"][0], c["bbox"][2]
        overlap = min(box[2], x1) - max(box[0], x0)
        if overlap <= 0:
            continue
        if not (overlap <= allowance and (x0 < box[0] < x1 <= box[2] or box[0] <= x0 < box[2] < x1)):
            return "overlaps_text"
    return "placeable"


def _part(span, chars):
    return {**span, "chars": chars, "text": "".join(c["c"] for c in chars), "bbox": _union(c["bbox"] for c in chars)}


def _insert(line, span):
    """Insert ``span`` into ``line`` at its x position, splitting a host span inside its whitespace run.

    Host characters go to the side of the span their centre is on. The whitespace run on each side is
    collapsed to one space; without whitespace, a space is added only for an empty gap wider than a word gap.
    """
    x0, x1 = span["bbox"][0], span["bbox"][2]
    centre = (x0 + x1) / 2
    left, right = [], []
    for s in line["spans"]:
        if s.get("_inserted"):
            (left if (s["bbox"][0] + s["bbox"][2]) / 2 < centre else right).append(s)
            continue
        chars = _chars(s)
        for side, part in ((left, [c for c in chars if (c["bbox"][0] + c["bbox"][2]) / 2 < centre]),
                           (right, [c for c in chars if (c["bbox"][0] + c["bbox"][2]) / 2 >= centre])):
            if part:
                side.append(_part(s, part))
    size = span.get("size") or (span["bbox"][3] - span["bbox"][1])

    def trim(parts, end):
        """Drop whitespace at the insertion side (``end`` -1 = last of left, 0 = first of right)."""
        found = False
        while parts:
            s = parts[end]
            if s.get("_inserted"):
                found = found or (s["text"][end] == " ")
                return found
            chars = _chars(s)
            while chars and _space(chars[end]["c"]):
                chars = chars[:-1] if end == -1 else chars[1:]
                found = True
            if chars:
                parts[end] = _part(s, chars)
                gap = x0 - parts[end]["bbox"][2] if end == -1 else parts[end]["bbox"][0] - x1
                return found or gap > GAP_SPACE * size
            parts.pop(end)
        return False

    before, after = trim(left, -1), trim(right, 0)
    text = span["text"].strip()
    moved = {**span, "text": (" " if before else "") + text + (" " if after else ""), "_inserted": True}
    line["spans"] = left + [moved] + right
    line["bbox"] = _union([line["bbox"], span["bbox"]])


def reorder(raw):
    """Return a copy of a PyMuPDF ``rawdict``/``dict`` page with late-inserted spans moved, and a report.

    Every span of the copy has ``text``; moved and unplaced spans are tagged with ``insertion``. The report
    lists ``moved`` spans with their host line and ``unplaced`` spans that overlay a line but were left in
    place for review. A page without such spans is returned unchanged apart from ``text``.
    """
    raw = copy.deepcopy(raw)
    lines = []
    for block in raw["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                span["text"] = _text(span)
            lines.append(line)
    moves, unplaced, unplaced_spans = [], [], []
    ink = {id(line): [c for s in line["spans"] if _visible(s) for c in _chars(s) if not _space(c["c"])]
           for line in lines}
    for line in lines:
        for span in line["spans"]:
            if not span["text"].strip() or not _visible(span):
                continue
            relations = [(host, r) for host in lines if host is not line
                         for r in [_relation(span, host, ink[id(host)])] if r]
            if not relations:
                continue
            record = {"text": span["text"].strip(), "bbox": [round(v, 2) for v in span["bbox"]]}
            placeable = [host for host, r in relations if r == "placeable"]
            if len(relations) == 1 and placeable:
                moves.append((span, line, placeable[0], record))
            else:
                reason = "ambiguous" if len(placeable) > 1 else "overlaps_text"
                hosts = ["".join(s["text"] for s in host["spans"]).strip() for host, _ in relations]
                unplaced.append({**record, "reason": reason, "hosts": hosts})
                unplaced_spans.append(span)
    # A span that is itself a host of another move stays put: chains are not reconstructed.
    hosts = {id(host) for _, _, host, _ in moves}
    blocked = [m for m in moves if id(m[1]) in hosts]
    for span, line, host, record in blocked:
        unplaced.append({**record, "reason": "chained", "hosts": ["".join(s["text"] for s in host["spans"]).strip()]})
        unplaced_spans.append(span)
    moves = [m for m in moves if id(m[1]) not in hosts]
    report = {"moved": [], "unplaced": unplaced}
    touched = set()
    for span, line, host, record in moves:
        line["spans"] = [s for s in line["spans"] if s is not span]
        _insert(host, span)
        touched.update((id(line), id(host)))
        report["moved"].append({**record, "host": "".join(s["text"] for s in host["spans"]).strip()})
    for span in (s for line in lines for s in line["spans"]):
        if span.pop("_inserted", None):
            span["insertion"] = "moved"
    for span in unplaced_spans:
        span["insertion"] = "unplaced"
    if touched:
        for block in raw["blocks"]:
            if block["type"] != 0 or not any(id(l) in touched for l in block["lines"]):
                continue
            block["lines"] = [l for l in block["lines"]
                              if id(l) not in touched or any(s["text"].strip() for s in l["spans"])]
            for l in block["lines"]:
                if id(l) in touched:
                    l["bbox"] = _union([s["bbox"] for s in l["spans"]])
            if block["lines"]:
                block["bbox"] = _union(l["bbox"] for l in block["lines"])
        raw["blocks"] = [b for b in raw["blocks"] if b["type"] != 0 or b["lines"]]
    return raw, report


def moved_spans(raw):
    return [s for b in raw["blocks"] if b["type"] == 0 for l in b["lines"] for s in l["spans"]
            if s.get("insertion") == "moved"]


def needs_review(lines):
    return any(s.get("insertion") == "unplaced" for l in lines for s in l["spans"])


def cell_text(raw, cell):
    """Text of a table cell from reordered lines, like PyMuPDF ``extract_cells`` (plain text).

    Characters count when more than half of their box is inside the cell; a moved span counts as a whole
    by its centre. Whitespace within a line is collapsed, matching ``Table.extract()`` word joining.
    """
    rect = pymupdf.Rect(cell)
    out = []
    for block in raw["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            if not rect.intersects(line["bbox"]):
                continue
            text = ""
            for span in line["spans"]:
                if span.get("insertion") == "moved":
                    x, y = (span["bbox"][0] + span["bbox"][2]) / 2, (span["bbox"][1] + span["bbox"][3]) / 2
                    text += span["text"] if rect.contains(pymupdf.Point(x, y)) else ""
                    continue
                for c in _chars(span):
                    box = pymupdf.Rect(c["bbox"])
                    if abs(box & rect) > 0.5 * abs(box):
                        text += c["c"]
                    elif _space(c["c"]):
                        text += " "
            if text.strip():
                out.append(" ".join(text.split()))
    return "\n".join(out)
