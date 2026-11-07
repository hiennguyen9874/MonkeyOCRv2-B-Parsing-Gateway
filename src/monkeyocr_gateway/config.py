"""Gateway settings read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Values of vlmExtraArgs.parsingMode (ADR-0001); "routed" is native-first page routing (ADR-0007).
PARSING_MODES = ("two-stage", "end-to-end", "routed")


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
    # Seconds allowed for the whole /health/ready check, as HPS_HEALTH_CHECK_TIMEOUT.
    health_check_timeout: float = 5.0
    # Retries of a vLLM call on 429/5xx/connection errors; the n-th retry waits
    # http_retry_backoff * 2**(n-1) seconds, capped at 30 s, as Upstream does.
    http_max_retries: int = 5
    http_retry_backoff: float = 1.0
    # Upstream --retry-repeat: re-ask recognition and end-to-end calls whose output
    # ends in repeated tokens, up to retry_repeat_max_retries times.
    retry_repeat: bool = False
    retry_repeat_max_retries: int = 3
    # /layout-parsing requests processed at once; more wait in a queue.
    request_concurrency: int = 16
    # vLLM chat calls in flight across the whole process.
    vllm_max_inflight: int = 64
    # Hosts that file URLs may point to (exact, case-insensitive); empty rejects every URL.
    url_allowlist: tuple[str, ...] = ()
    # Seconds allowed to download a file URL, redirects included.
    url_fetch_timeout: float = 30.0
    server_backend: str = "vllm"

    def __post_init__(self) -> None:
        if self.server_backend not in ("vllm", "llama-cpp"):
            raise ValueError(
                f"MOCR2_SERVER_BACKEND must be 'vllm' or 'llama-cpp', got {self.server_backend!r}"
            )
        for name, value in [
            ("MOCR2_PAGE_CONCURRENCY", self.page_concurrency),
            ("MOCR2_REQUEST_CONCURRENCY", self.request_concurrency),
            ("MOCR2_SERVER_MAX_INFLIGHT", self.vllm_max_inflight),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.http_max_retries < 0 or self.retry_repeat_max_retries < 0:
            raise ValueError("MOCR2_HTTP_MAX_RETRIES and MOCR2_REC_MAX_RETRIES must not be negative")

    @classmethod
    def from_env(cls) -> Settings:
        # Names and defaults follow Upstream's parsing/fastapi/main.py. Upstream
        # defaults to two-stage (end2end=False), as does the validated legal-vi-ocr run.
        default_parsing_mode = os.getenv("MOCR2_PARSING_MODE", "two-stage")
        if default_parsing_mode not in PARSING_MODES:
            raise ValueError(
                f"MOCR2_PARSING_MODE must be 'two-stage', 'end-to-end' or 'routed', "
                f"got {default_parsing_mode!r}"
            )
        return cls(
            vllm_url=os.getenv("MOCR2_SERVER_URL", "http://127.0.0.1:8000"),
            served_model_name=os.getenv("MOCR2_SERVED_MODEL_NAME", "MonkeyOCRv2"),
            max_pixels=int(os.getenv("MOCR2_MAX_PIXELS", "1003520")),
            request_timeout=float(os.getenv("MOCR2_REQUEST_TIMEOUT", "300")),
            default_parsing_mode=default_parsing_mode,
            server_backend=os.getenv("MOCR2_SERVER_BACKEND", "vllm"),
            page_concurrency=int(os.getenv("MOCR2_PAGE_CONCURRENCY", "8")),
            max_pages=int(os.getenv("MOCR2_MAX_PAGES", "0")),
            max_body_bytes=int(os.getenv("MOCR2_MAX_BODY_BYTES", str(100 * 1024 * 1024))),
            health_check_timeout=float(os.getenv("MOCR2_HEALTH_CHECK_TIMEOUT", "5")),
            http_max_retries=int(os.getenv("MOCR2_HTTP_MAX_RETRIES", "5")),
            http_retry_backoff=float(os.getenv("MOCR2_HTTP_RETRY_BACKOFF", "1.0")),
            retry_repeat=os.getenv("MOCR2_RETRY_REPEAT", "0").lower() in {"1", "true", "yes"},
            retry_repeat_max_retries=int(os.getenv("MOCR2_REC_MAX_RETRIES", "3")),
            request_concurrency=int(os.getenv("MOCR2_REQUEST_CONCURRENCY", "16")),
            vllm_max_inflight=int(os.getenv("MOCR2_SERVER_MAX_INFLIGHT", "64")),
            url_allowlist=tuple(
                host.strip().lower()
                for host in os.getenv("MOCR2_URL_ALLOWLIST", "").split(",")
                if host.strip()
            ),
            url_fetch_timeout=float(os.getenv("MOCR2_URL_FETCH_TIMEOUT", "30")),
        )
