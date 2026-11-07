"""HTTP API compatible with PaddleOCR-VL `/layout-parsing`.

Run with: uv run uvicorn --factory monkeyocr_gateway.app:create_app
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, FiniteFloat

from .config import PARSING_MODES, Settings
from .document_loader import DocumentError, open_document, read_file
from .paddle_response import build_page_result, DEFAULT_MARKDOWN_IGNORE_LABELS
from .page_parser import gather_or_cancel, parse_page
from .routed import NativeDocument, PAGE_ROUTER_SETTINGS, parse_routed_image, parse_routed_page
from .vllm_client import GenerationOptions, VLLMClient, VLLMError

# Preprocessing changes the coordinate space of the result, so it is rejected
# rather than ignored (ADR-0004).
PREPROCESSING_FLAGS = ("useDocOrientationClassify", "useDocUnwarping")


class LayoutParsingRequest(BaseModel):
    # Other PaddleOCR-VL fields (seal, chart, restructurePages, minPixels/maxPixels…)
    # are accepted and not applied (ADR-0002).
    model_config = ConfigDict(extra="allow")

    file: str
    fileType: int | None = None
    logId: str | None = None
    vlmExtraArgs: dict | None = None
    markdownIgnoreLabels: list[str] | None = None
    returnMarkdownImages: bool = True
    useDocOrientationClassify: bool | None = None
    useDocUnwarping: bool | None = None
    temperature: FiniteFloat | None = None
    topP: FiniteFloat | None = None
    repetitionPenalty: FiniteFloat | None = None
    maxNewTokens: int | None = None


def _model_settings(markdown_ignore_labels: list[str], mode: str) -> dict:
    """What the gateway actually does, whatever the request asked for (ADR-0002)."""
    settings = {
        "use_doc_preprocessor": False,
        "use_layout_detection": True,
        "use_chart_recognition": False,
        "use_seal_recognition": False,
        "use_ocr_for_image_block": False,
        "format_block_content": False,
        "merge_layout_blocks": False,
        "restructure_pages": False,
        "markdown_ignore_labels": markdown_ignore_labels,
        "parsing_mode": mode,
    }
    if mode == "routed":
        settings["page_router"] = PAGE_ROUTER_SETTINGS
    return settings


def _new_log_id() -> str:
    return str(uuid.uuid4())


def _error_response(log_id: str, status_code: int, message: str) -> JSONResponse:
    # As PaddleOCR-VL serving, an error carries no result.
    return JSONResponse(
        status_code=status_code,
        content={"logId": log_id, "errorCode": status_code, "errorMsg": message},
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
        # Requests beyond the limit wait here, in arrival order.
        app.state.request_slots = asyncio.Semaphore(settings.request_concurrency)
        try:
            yield
        finally:
            await app.state.vllm.aclose()

    app = FastAPI(title="MonkeyOCRv2-B-Parsing Gateway", lifespan=lifespan)
    app.add_middleware(BodySizeLimit, max_bytes=settings.max_body_bytes)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        # Keep the client's logId when the body carries one, as for other errors.
        log_id = exc.body.get("logId") if isinstance(exc.body, dict) else None
        if not isinstance(log_id, str) or not log_id:
            log_id = _new_log_id()
        message = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'][1:]) or 'body'}: {error['msg']}"
            for error in exc.errors()
        )
        return _error_response(log_id, 422, f"Invalid request: {message}")

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
        for flag in PREPROCESSING_FLAGS:
            if getattr(body, flag):
                return _error_response(
                    log_id,
                    422,
                    f"{flag}=true is not supported: document preprocessing would change the "
                    "result coordinate space; send false or null",
                )
        mode = (body.vlmExtraArgs or {}).get("parsingMode")
        if mode is None:
            mode = settings.default_parsing_mode
        elif mode not in PARSING_MODES:
            return _error_response(
                log_id,
                422,
                f"Invalid vlmExtraArgs.parsingMode {mode!r}; expected 'two-stage', 'end-to-end' or 'routed'",
            )
        async with app.state.request_slots:
            try:
                return await _parse_document(body, log_id, mode)
            except (DocumentError, VLLMError) as exc:
                return _error_response(log_id, exc.status_code, str(exc))

    async def _parse_document(body: LayoutParsingRequest, log_id: str, mode: str) -> dict:
        """Raises DocumentError or VLLMError; any page or Khối failing fails the request."""
        data = await read_file(
            body.file, settings.url_allowlist, settings.max_body_bytes, settings.url_fetch_timeout
        )
        document = await asyncio.to_thread(open_document, data, body.fileType, settings.max_pages)

        markdown_ignore_labels = body.markdownIgnoreLabels
        if markdown_ignore_labels is None:
            markdown_ignore_labels = DEFAULT_MARKDOWN_IGNORE_LABELS
        model_settings = _model_settings(markdown_ignore_labels, mode)
        options = GenerationOptions(
            temperature=body.temperature,
            top_p=body.topP,
            repetition_penalty=body.repetitionPenalty,
            max_new_tokens=body.maxNewTokens,
        )
        # As in PaddleOCR-VL, page_count is set for PDF input only.
        page_count = document.page_count if document.kind == "pdf" else None
        semaphore = asyncio.Semaphore(settings.page_concurrency)
        repeat_retries = settings.retry_repeat_max_retries if settings.retry_repeat else 0
        native = None

        async def parse(index: int):
            async with semaphore:
                image = await asyncio.to_thread(document.page, index)
                routing = None
                if native is not None:
                    plan = await asyncio.to_thread(native.plan, index)
                    blocks, routing = await parse_routed_page(
                        image, plan, app.state.vllm, options, repeat_retries
                    )
                elif mode == "routed":
                    blocks, routing = await parse_routed_image(
                        image, app.state.vllm, options, repeat_retries
                    )
                else:
                    blocks = await parse_page(image, mode, app.state.vllm, options, repeat_retries)
                # Crop and encode before releasing the page concurrency slot.
                return await asyncio.to_thread(
                    build_page_result,
                    image,
                    blocks,
                    model_settings,
                    page_count,
                    index,
                    body.returnMarkdownImages,
                    routing,
                )

        try:
            if mode == "routed" and document.kind == "pdf":
                # The embedded text layer is read with PyMuPDF (ADR-0007).
                native = await asyncio.to_thread(NativeDocument, data)
                if native.page_count != document.page_count:
                    raise DocumentError(
                        422, "file is not a readable PDF: page count differs between PDF readers"
                    )
            pages = await gather_or_cancel([parse(index) for index in range(document.page_count)])
        finally:
            await asyncio.to_thread(document.close)
            if native is not None:
                await asyncio.to_thread(native.close)

        if document.kind == "pdf":
            data_info = {
                "type": "pdf",
                "numPages": document.page_count,
                "pages": [
                    {
                        "width": page["prunedResult"]["width"],
                        "height": page["prunedResult"]["height"],
                    }
                    for page in pages
                ],
            }
        else:
            # A multi-frame TIFF is described by its first frame.
            first_page = pages[0]["prunedResult"]
            data_info = {
                "type": "image",
                "width": first_page["width"],
                "height": first_page["height"],
            }
        return {
            "logId": log_id,
            "errorCode": 0,
            "errorMsg": "Success",
            "result": {
                "layoutParsingResults": pages,
                "dataInfo": data_info,
            },
        }

    return app
