"""HTML for native tables.

Ported from legal-vi-ocr `src/legal_vi_ocr/views.py` at 6e7ad59 (`cell_display`, `_table`
as `table_html`). Page views are not ported: the gateway answers in PaddleOCR-VL format.
"""

from html import escape


def cell_display(text):
    """Cell text for reading: lines joined with a space, or directly after a line-final hyphen
    (``2028-`` + ``2030``, ``UBND-`` + ``VHXH``). The canonical JSON keeps the original lines."""
    out = ""
    for line in (l.strip() for l in text.split("\n")):
        if not line:
            continue
        glue = "" if not out or out.endswith("-") else " "
        out += glue + line
    return out


def table_html(table):
    cells = {(cell["row"], cell["column"]): cell for cell in table["cells"]}
    rows = []
    for row in range(table["rows"]):
        parts = []
        for col in range(table["columns"]):
            cell = cells.get((row, col))
            if cell is None:
                continue
            tag = "th" if cell["header"] else "td"
            parts.append(f'<{tag} rowspan="{cell["rowspan"]}" colspan="{cell["colspan"]}">'
                         f'{escape(cell_display(cell["text"]))}</{tag}>')
        rows.append("<tr>" + "".join(parts) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"
