"""HTTP API compatible with PaddleOCR-VL `/layout-parsing`.

Run with: uv run uvicorn --factory monkeyocr_gateway.app:create_app
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .config import PARSING_MODES, Settings
from .document_loader import load_pages
from .page_parser import parse_page
from .paddle_response import build_page_result
from .vllm_client import VLLMClient


class LayoutParsingRequest(BaseModel):
    # Other PaddleOCR-VL fields are accepted and ignored.
    model_config = ConfigDict(extra="allow")

    file: str
    fileType: int | None = None
    logId: str | None = None
    vlmExtraArgs: dict | None = None


def _new_log_id() -> str:
    return str(uuid.uuid4())


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.vllm = VLLMClient(settings)
        try:
            yield
        finally:
            await app.state.vllm.aclose()

    app = FastAPI(title="MonkeyOCRv2-B-Parsing Gateway", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"logId": _new_log_id(), "errorCode": 0, "errorMsg": "Healthy"}

    @app.post("/layout-parsing")
    async def layout_parsing(body: LayoutParsingRequest):
        log_id = body.logId or _new_log_id()
        mode = (body.vlmExtraArgs or {}).get("parsingMode")
        if mode is None:
            mode = settings.default_parsing_mode
        elif mode not in PARSING_MODES:
            return JSONResponse(
                status_code=422,
                content={
                    "logId": log_id,
                    "errorCode": 422,
                    "errorMsg": (
                        f"Invalid vlmExtraArgs.parsingMode {mode!r}; "
                        "expected 'two-stage' or 'end-to-end'"
                    ),
                    "result": None,
                },
            )
        pages = await asyncio.to_thread(load_pages, body.file)
        page = pages[0]
        blocks = await parse_page(page, mode, app.state.vllm)
        width, height = page.size
        model_settings = {"parsing_mode": mode}
        return {
            "logId": log_id,
            "errorCode": 0,
            "errorMsg": "Success",
            "result": {
                "layoutParsingResults": [build_page_result(width, height, blocks, model_settings)],
                "dataInfo": {"type": "image", "width": width, "height": height},
            },
        }

    return app
