"""Canonical page contract and validation, independent of any engine.

Ported verbatim from legal-vi-ocr `src/legal_vi_ocr/canonical.py` at 6e7ad59. The gateway
uses it only for the native adapters' internal output; responses stay PaddleOCR-VL.
"""

import json
import math
import re


def validate_page(page):
    def check(condition, message):
        if not condition:
            raise ValueError(message)

    def obj(value, fields, name):
        check(isinstance(value, dict) and set(value) == set(fields), f"invalid {name}")

    def integer(value, minimum=0):
        return type(value) is int and value >= minimum

    def number(value):
        return type(value) in (int, float) and math.isfinite(value)

    def provenance(value):
        obj(value, ("engine", "version", "model", "revision", "config"), "provenance")
        check(
            all(isinstance(value[k], str) and value[k] for k in ("engine", "version")),
            "invalid engine or version",
        )
        check(
            all(value[k] is None or isinstance(value[k], str) for k in ("model", "revision"))
            and isinstance(value["config"], dict),
            "invalid provenance",
        )

    def quality(value):
        obj(value, ("confidence", "truncated", "review"), "quality")
        check(
            value["confidence"] is None
            or (number(value["confidence"]) and 0 <= value["confidence"] <= 1),
            "invalid confidence",
        )
        check(
            type(value["truncated"]) is bool and type(value["review"]) is bool,
            "invalid quality flags",
        )

    def bbox(value):
        check(
            isinstance(value, list)
            and len(value) == 4
            and all(number(x) for x in value)
            and value[0] <= value[2]
            and value[1] <= value[3],
            "invalid bbox",
        )

    obj(
        page,
        ("schema_version", "file", "documents", "page", "blocks", "provenance", "quality"),
        "page",
    )
    check(page["schema_version"] == "1.0", "unsupported schema version")
    obj(page["file"], ("path", "sha256"), "file")
    check(
        isinstance(page["file"]["path"], str)
        and bool(page["file"]["path"])
        and isinstance(page["file"]["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", page["file"]["sha256"]) is not None,
        "invalid file",
    )
    check(isinstance(page["blocks"], list), "invalid blocks")
    check(isinstance(page["documents"], list), "invalid documents")
    for doc in page["documents"]:
        obj(doc, ("id", "boundaries", "legal_units"), "document")
        check(
            isinstance(doc["id"], str)
            and bool(doc["id"])
            and isinstance(doc["boundaries"], list)
            and isinstance(doc["legal_units"], list),
            "invalid document",
        )
        for boundary in doc["boundaries"]:
            obj(boundary, ("page", "block"), "boundary")
            check(integer(boundary["page"], 1) and integer(boundary["block"]), "invalid boundary")
        for unit in doc["legal_units"]:
            obj(unit, ("id", "type", "label", "parent", "blocks"), "legal unit")
            check(
                all(isinstance(unit[k], str) for k in ("id", "type", "label"))
                and (unit["parent"] is None or isinstance(unit["parent"], str))
                and isinstance(unit["blocks"], list)
                and all(integer(i) and i < len(page["blocks"]) for i in unit["blocks"]),
                "invalid legal unit",
            )
    obj(
        page["page"],
        ("number", "width", "height", "coordinate_unit", "type", "routing"),
        "page metadata",
    )
    meta = page["page"]
    check(
        integer(meta["number"], 1)
        and number(meta["width"])
        and meta["width"] > 0
        and number(meta["height"])
        and meta["height"] > 0
        and meta["coordinate_unit"] == "pt"
        and (meta["type"] is None or isinstance(meta["type"], str))
        and (meta["routing"] is None or isinstance(meta["routing"], dict)),
        "invalid page metadata",
    )
    routing = meta["routing"]
    if routing is not None:
        fields = (
            "decision",
            "reasons",
            "signals",
            "thresholds",
            "near_blank",
            "supplement_regions",
            "requests",
        )
        # Older immutable runs have only the original routing fields.
        optional = (
            "native_fallback",
            "effective_mode",
            "review_reasons",
            "retained_regions",
            "table_fallbacks",
        )
        obj(routing, (*fields, *(k for k in optional if k in routing)), "routing")
        if "effective_mode" in routing:
            check(
                routing["effective_mode"]
                in ("native-only", "native+ocr", "full-ocr", "near-blank-review"),
                "invalid routing effective mode",
            )
        if "review_reasons" in routing:
            check(
                isinstance(routing["review_reasons"], list)
                and all(
                    isinstance(r, dict) and isinstance(r.get("code"), str) and r["code"]
                    for r in routing["review_reasons"]
                ),
                "invalid routing review reasons",
            )
        if "table_fallbacks" in routing:
            check(
                isinstance(routing["table_fallbacks"], list)
                and all(isinstance(r, dict) for r in routing["table_fallbacks"]),
                "invalid table fallbacks",
            )
        if "retained_regions" in routing:
            check(isinstance(routing["retained_regions"], list), "invalid retained regions")
            for region in routing["retained_regions"]:
                check(
                    isinstance(region, dict) and isinstance(region.get("reason"), str),
                    "invalid retained region",
                )
                bbox(region.get("bbox"))
        check(
            routing["decision"] in ("native", "ocr")
            and isinstance(routing["reasons"], list)
            and bool(routing["reasons"])
            and all(isinstance(r, str) and bool(r) for r in routing["reasons"]),
            "invalid routing decision/reasons",
        )
        check(
            isinstance(routing["signals"], dict)
            and isinstance(routing["thresholds"], dict)
            and type(routing["near_blank"]) is bool
            and isinstance(routing["supplement_regions"], list)
            and isinstance(routing["requests"], list),
            "invalid routing evidence",
        )
        if "native_fallback" in routing:
            check(isinstance(routing["native_fallback"], str), "invalid native fallback")
        for region in routing["supplement_regions"]:
            bbox(region)
        for request in routing["requests"]:
            check(
                isinstance(request, dict)
                and request.get("status") in ("pending", "ok", "error")
                and isinstance(request.get("input"), dict),
                "invalid routing request",
            )
    provenance(page["provenance"])
    quality(page["quality"])
    for block in page["blocks"]:
        keys = ("type", "bbox", "coordinate_unit", "raw_text", "provenance", "quality")
        check(isinstance(block, dict), "invalid block")
        obj(block, (*keys, "table") if block.get("type") == "table" else keys, "block")
        check(
            isinstance(block["type"], str)
            and bool(block["type"])
            and block["coordinate_unit"] == "pt"
            and isinstance(block["raw_text"], str),
            "invalid block",
        )
        bbox(block["bbox"])
        provenance(block["provenance"])
        quality(block["quality"])
        if block["type"] == "table":
            table = block["table"]
            obj(table, ("rows", "columns", "cells", "links"), "table")
            check(
                integer(table["rows"], 1)
                and integer(table["columns"], 1)
                and isinstance(table["cells"], list)
                and isinstance(table["links"], list),
                "invalid table",
            )
            for link in table["links"]:
                obj(link, ("page", "block", "review"), "table link")
                check(
                    integer(link["page"], 1)
                    and integer(link["block"])
                    and type(link["review"]) is bool,
                    "invalid table link",
                )
            occupied = set()
            for cell in table["cells"]:
                obj(cell, ("row", "column", "rowspan", "colspan", "header", "text"), "cell")
                check(
                    integer(cell["row"])
                    and integer(cell["column"])
                    and integer(cell["rowspan"], 1)
                    and integer(cell["colspan"], 1)
                    and cell["row"] + cell["rowspan"] <= table["rows"]
                    and cell["column"] + cell["colspan"] <= table["columns"]
                    and type(cell["header"]) is bool
                    and isinstance(cell["text"], str),
                    "invalid cell",
                )
                positions = {
                    (r, c)
                    for r in range(cell["row"], cell["row"] + cell["rowspan"])
                    for c in range(cell["column"], cell["column"] + cell["colspan"])
                }
                check(not positions & occupied, "overlapping cells")
                occupied.update(positions)
            check(len(occupied) == table["rows"] * table["columns"], "missing table cells")
    try:
        json.dumps(page, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("page is not valid JSON") from exc
