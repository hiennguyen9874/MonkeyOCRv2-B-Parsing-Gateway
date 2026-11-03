"""Gateway settings read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    vllm_url: str
    served_model_name: str
    max_pixels: int
    request_timeout: float

    @classmethod
    def from_env(cls) -> Settings:
        # Names and defaults follow Upstream's parsing/fastapi/main.py.
        return cls(
            vllm_url=os.getenv("MOCR2_SERVER_URL", "http://127.0.0.1:8000"),
            served_model_name=os.getenv("MOCR2_SERVED_MODEL_NAME", "MonkeyOCRv2"),
            max_pixels=int(os.getenv("MOCR2_MAX_PIXELS", "1003520")),
            request_timeout=float(os.getenv("MOCR2_REQUEST_TIMEOUT", "300")),
        )
