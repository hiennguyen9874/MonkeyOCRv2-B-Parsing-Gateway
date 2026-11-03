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
from .document_loader import DocumentError, open_document
from .page_parser import gather_or_cancel, parse_page
from .paddle_response import DEFAULT_MARKDOWN_IGNORE_LABELS, build_page_result
from .vllm_client import VLLMClient


class LayoutParsingRequest(BaseModel):
    # Other PaddleOCR-VL fields are accepted and ignored.
    model_config = ConfigDict(extra="allow")

    file: str
    fileType: int | None = None
    logId: str | None = None
    vlmExtraArgs: dict | None = None
    markdownIgnoreLabels: list[str] | None = None
    returnMarkdownImages: bool = True


def _new_log_id() -> str:
    return str(uuid.uuid4())


def _error_response(log_id: str, status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"logId": log_id, "errorCode": status_code, "errorMsg": message, "result": None},
    )


class BodySizeLimit:
    """ASGI middleware: 413 for bodies over max_bytes, with or without Content-Length."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def _reject(self, scope, receive, send) -> None:
        message = f"Request body exceeds {self.max_bytes} bytes (MOCR2_MAX_BODY_BYTES)"
        await _error_response(_new_log_id(), 413, message)(scope, receive, send)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.max_bytes:
            await self.app(scope, receive, send)
            return
        content_length = dict(scope["headers"]).get(b"content-length")
        if content_length is not None and int(content_length) > self.max_bytes:
            await self._reject(scope, receive, send)
            return

        received = 0
        rejected = False

        async def limited_receive():
            nonlocal received, rejected
            message = await receive()
            if message["type"] == "http.request" and not rejected:
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    rejected = True
                    await self._reject(scope, receive, send)
                    # The app sees a disconnect and stops reading.
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            if not rejected:
                await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            # The app failing on the cut-off body is expected once the 413 is sent.
            if not rejected:
                raise


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
    app.add_middleware(BodySizeLimit, max_bytes=settings.max_body_bytes)

    @app.get("/health")
    async def health():
        return {"logId": _new_log_id(), "errorCode": 0, "errorMsg": "Healthy"}

    @app.get("/health/ready")
    async def ready():
        reason = await app.state.vllm.check_ready()
        if reason is not None:
            return JSONResponse(
                status_code=503,
                content={"logId": _new_log_id(), "errorCode": 503, "errorMsg": reason},
            )
        return {"logId": _new_log_id(), "errorCode": 0, "errorMsg": "Ready"}

    @app.post("/layout-parsing")
    async def layout_parsing(body: LayoutParsingRequest):
        log_id = body.logId or _new_log_id()
        mode = (body.vlmExtraArgs or {}).get("parsingMode")
        if mode is None:
            mode = settings.default_parsing_mode
        elif mode not in PARSING_MODES:
            return _error_response(
                log_id,
                422,
                f"Invalid vlmExtraArgs.parsingMode {mode!r}; expected 'two-stage' or 'end-to-end'",
            )
        try:
            document = await asyncio.to_thread(
                open_document, body.file, body.fileType, settings.max_pages
            )
        except DocumentError as exc:
            return _error_response(log_id, exc.status_code, str(exc))

        semaphore = asyncio.Semaphore(settings.page_concurrency)

        async def parse(index: int):
            async with semaphore:
                image = await asyncio.to_thread(document.page, index)
                return image.size, await parse_page(image, mode, app.state.vllm)

        try:
            pages = await gather_or_cancel([parse(index) for index in range(document.page_count)])
        except DocumentError as exc:
            return _error_response(log_id, exc.status_code, str(exc))
        finally:
            await asyncio.to_thread(document.close)

        markdown_ignore_labels = body.markdownIgnoreLabels
        if markdown_ignore_labels is None:
            markdown_ignore_labels = DEFAULT_MARKDOWN_IGNORE_LABELS
        model_settings = {"markdown_ignore_labels": markdown_ignore_labels, "parsing_mode": mode}
        # As in PaddleOCR-VL, page_count is set for PDF input only.
        page_count = document.page_count if document.kind == "pdf" else None
        if document.kind == "pdf":
            data_info = {
                "type": "pdf",
                "numPages": document.page_count,
                "pages": [{"width": width, "height": height} for (width, height), _ in pages],
            }
        else:
            # A multi-frame TIFF is described by its first frame.
            width, height = pages[0][0]
            data_info = {"type": "image", "width": width, "height": height}
        # Off the event loop: Picture crops are JPEG-encoded here.
        layout_parsing_results = await asyncio.to_thread(
            lambda: [
                build_page_result(
                    width, height, blocks, model_settings, page_count, body.returnMarkdownImages
                )
                for (width, height), blocks in pages
            ]
        )
        return {
            "logId": log_id,
            "errorCode": 0,
            "errorMsg": "Success",
            "result": {
                "layoutParsingResults": layout_parsing_results,
                "dataInfo": data_info,
            },
        }

    return app
