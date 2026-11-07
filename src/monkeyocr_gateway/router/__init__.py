"""Native-first page router (Chế độ định tuyến theo trang, ADR-0007).

Ported from the sibling legal-vi-ocr repo at revision `6e7ad59`
(`docs/page-router-architecture.md`, issue #12 there). Each module names the
source file it was ported from. The helpers are kept as close to the source as
the gateway allows, so `tests/test_router_*.py` can reuse the source tests.

Only CPU dependencies: PyMuPDF and NumPy (ADR-0003 stays intact).
"""

SOURCE_REVISION = "legal-vi-ocr@6e7ad59"
