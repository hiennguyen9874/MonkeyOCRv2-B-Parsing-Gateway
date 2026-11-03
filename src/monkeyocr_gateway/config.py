"""Gateway settings read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Values of vlmExtraArgs.parsingMode (ADR-0001).
PARSING_MODES = ("two-stage", "end-to-end")


@dataclass(frozen=True)
class Settings:
    vllm_url: str
    served_model_name: str
    max_pixels: int
    request_timeout: float
    default_parsing_mode: str
    # Pages of one request parsed in parallel.
    page_concurrency: int = 8
    # 0 means no limit.
    max_pages: int = 0
    max_body_bytes: int = 100 * 1024 * 1024

    @classmethod
    def from_env(cls) -> Settings:
        # Names and defaults follow Upstream's parsing/fastapi/main.py. Upstream
        # defaults to two-stage (end2end=False), as does the validated legal-vi-ocr run.
        default_parsing_mode = os.getenv("MOCR2_PARSING_MODE", "two-stage")
        if default_parsing_mode not in PARSING_MODES:
            raise ValueError(
                f"MOCR2_PARSING_MODE must be 'two-stage' or 'end-to-end', got {default_parsing_mode!r}"
            )
        return cls(
            vllm_url=os.getenv("MOCR2_SERVER_URL", "http://127.0.0.1:8000"),
            served_model_name=os.getenv("MOCR2_SERVED_MODEL_NAME", "MonkeyOCRv2"),
            max_pixels=int(os.getenv("MOCR2_MAX_PIXELS", "1003520")),
            request_timeout=float(os.getenv("MOCR2_REQUEST_TIMEOUT", "300")),
            default_parsing_mode=default_parsing_mode,
            page_concurrency=int(os.getenv("MOCR2_PAGE_CONCURRENCY", "8")),
            max_pages=int(os.getenv("MOCR2_MAX_PAGES", "0")),
            max_body_bytes=int(os.getenv("MOCR2_MAX_BODY_BYTES", str(100 * 1024 * 1024))),
        )
