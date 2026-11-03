"""Turn the request `file` into page images."""

from __future__ import annotations

import asyncio
import base64
import binascii
import threading
from io import BytesIO

import httpx
import pypdfium2 as pdfium
from PIL import Image

from .upstream import _apply_jpeg_orientation

PDF_FILE_TYPE = 0
IMAGE_FILE_TYPE = 1
# Upstream core_runner._render_pdf_page renders PDF pages at 200 DPI.
PDF_RENDER_SCALE = 200 / 72
# PDFium is not thread-safe (Upstream core_runner.PDFIUM_LOCK); every call takes this lock.
PDFIUM_LOCK = threading.RLock()
# Redirects followed when downloading a file URL; every hop must be allowlisted.
MAX_URL_REDIRECTS = 5


class DocumentError(Exception):
    """Input the gateway cannot accept; status_code is the HTTP status to return."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class PdfDocument:
    """Pages are rendered on demand so a long PDF is never held in memory at once."""

    kind = "pdf"

    def __init__(self, data: bytes):
        with PDFIUM_LOCK:
            self._pdf = pdfium.PdfDocument(data)
            self.page_count = len(self._pdf)

    def page(self, index: int) -> Image.Image:
        # Upstream core_runner._render_pdf_page at 106914e.
        with PDFIUM_LOCK:
            try:
                page = self._pdf[index]
                try:
                    bitmap = page.render(scale=PDF_RENDER_SCALE)
                    try:
                        return bitmap.to_pil().convert("RGB")
                    finally:
                        bitmap.close()
                finally:
                    page.close()
            except pdfium.PdfiumError as exc:
                raise DocumentError(422, f"file is not a readable PDF: {exc}") from None

    def close(self) -> None:
        with PDFIUM_LOCK:
            self._pdf.close()


class ImageDocument:
    """One page per image; a multi-frame TIFF gives one page per frame."""

    kind = "image"

    def __init__(self, data: bytes):
        self._image = Image.open(BytesIO(data))
        self.page_count = self._image.n_frames if self._image.format == "TIFF" else 1
        # Frames share one file handle, so seek + decode must not interleave.
        self._lock = threading.Lock()

    def page(self, index: int) -> Image.Image:
        with self._lock:
            try:
                self._image.seek(index)
                return _apply_jpeg_orientation(self._image).convert("RGB")
            except (OSError, EOFError, Image.DecompressionBombError) as exc:
                raise DocumentError(422, f"file is not a readable image: {exc}") from None

    def close(self) -> None:
        with self._lock:
            self._image.close()


async def read_file(
    file: str, url_allowlist: tuple[str, ...], max_bytes: int, fetch_timeout: float
) -> bytes:
    """The bytes of the request `file`: a Base64 string, or an http(s) URL whose host
    is in url_allowlist. max_bytes (0 = no limit) caps a download.

    Raises DocumentError for input the gateway cannot accept.
    """
    # Base64 has no ':', so a scheme separator near the start means a URL.
    if "://" in file[:16]:
        return await _download(file, url_allowlist, max_bytes, fetch_timeout)
    return await asyncio.to_thread(_decode_base64, file)


def _decode_base64(file: str) -> bytes:
    try:
        data = base64.b64decode(file, validate=True)
    except (binascii.Error, ValueError):
        raise DocumentError(422, "file is not valid Base64") from None
    if not data:
        raise DocumentError(422, "file is empty after Base64 decoding")
    return data


def _check_url(url: httpx.URL, allowlist: tuple[str, ...]) -> None:
    if url.scheme not in ("http", "https"):
        raise DocumentError(422, f"file URL scheme {url.scheme!r} is not supported; use http or https")
    if not allowlist:
        raise DocumentError(422, "file URLs are not accepted: MOCR2_URL_ALLOWLIST is empty; send Base64")
    if url.host.lower() not in allowlist:
        raise DocumentError(422, f"file URL host {url.host!r} is not in MOCR2_URL_ALLOWLIST")


async def _download(file: str, allowlist: tuple[str, ...], max_bytes: int, timeout: float) -> bytes:
    """GET the URL, following redirects by hand so that every hop is checked against
    the allowlist. The timeout covers the whole download, redirects included."""
    try:
        url = httpx.URL(file)
    except httpx.InvalidURL as exc:
        raise DocumentError(422, f"file is not a valid URL: {exc}") from None
    try:
        async with asyncio.timeout(timeout), httpx.AsyncClient(timeout=timeout) as client:
            for _ in range(MAX_URL_REDIRECTS + 1):
                _check_url(url, allowlist)
                async with client.stream("GET", url) as response:
                    if response.next_request is not None:
                        url = response.next_request.url
                        continue
                    if response.status_code != 200:
                        raise DocumentError(
                            422, f"Downloading the file URL failed: HTTP {response.status_code}"
                        )
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data += chunk
                        if max_bytes and len(data) > max_bytes:
                            raise DocumentError(
                                413, f"file URL content exceeds {max_bytes} bytes (MOCR2_MAX_BODY_BYTES)"
                            )
                    if not data:
                        raise DocumentError(422, "file URL returned no content")
                    return bytes(data)
            raise DocumentError(422, f"file URL redirected more than {MAX_URL_REDIRECTS} times")
    except (TimeoutError, httpx.TimeoutException):
        raise DocumentError(
            504, f"Downloading the file URL took longer than {timeout}s (MOCR2_URL_FETCH_TIMEOUT)"
        ) from None
    except httpx.HTTPError as exc:
        raise DocumentError(422, f"Downloading the file URL failed: {type(exc).__name__}: {exc}") from None


def open_document(data: bytes, file_type: int | None, max_pages: int | None) -> PdfDocument | ImageDocument:
    """Open the file bytes; fileType 0 is PDF, 1 is image, None detects by magic bytes.

    Raises DocumentError for input the gateway cannot accept.
    """
    if file_type not in (PDF_FILE_TYPE, IMAGE_FILE_TYPE, None):
        raise DocumentError(422, f"Invalid fileType {file_type!r}; expected 0 (PDF), 1 (image) or null")

    detected = file_type is None
    if detected:
        # Same PDF check as Upstream's gradio demo.
        file_type = PDF_FILE_TYPE if b"%PDF-" in data[:1024] else IMAGE_FILE_TYPE
    try:
        document = PdfDocument(data) if file_type == PDF_FILE_TYPE else ImageDocument(data)
    except (pdfium.PdfiumError, OSError, Image.DecompressionBombError) as exc:
        if file_type == PDF_FILE_TYPE:
            message = f"file is not a readable PDF: {exc}"
        elif detected:
            message = "Unrecognized file type; expected a PDF or an image (PNG, JPEG, BMP, WEBP, TIFF)"
        else:
            message = f"file is not a readable image: {exc}"
        raise DocumentError(422, message) from None

    if max_pages and document.page_count > max_pages:
        document.close()
        raise DocumentError(
            413, f"Document has {document.page_count} pages; the limit is {max_pages} (MOCR2_MAX_PAGES)"
        )
    return document
