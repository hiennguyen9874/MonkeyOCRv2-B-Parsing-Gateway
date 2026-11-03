"""Turn the request `file` into page images."""

from __future__ import annotations

import base64
import binascii
import threading
from io import BytesIO

import pypdfium2 as pdfium
from PIL import Image

from .upstream import _apply_jpeg_orientation

PDF_FILE_TYPE = 0
IMAGE_FILE_TYPE = 1
# Upstream core_runner._render_pdf_page renders PDF pages at 200 DPI.
PDF_RENDER_SCALE = 200 / 72
# PDFium is not thread-safe (Upstream core_runner.PDFIUM_LOCK); every call takes this lock.
PDFIUM_LOCK = threading.RLock()


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
            page = self._pdf[index]
            try:
                bitmap = page.render(scale=PDF_RENDER_SCALE)
                try:
                    return bitmap.to_pil().convert("RGB")
                finally:
                    bitmap.close()
            finally:
                page.close()

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
            self._image.seek(index)
            return _apply_jpeg_orientation(self._image).convert("RGB")

    def close(self) -> None:
        with self._lock:
            self._image.close()


def open_document(file: str, file_type: int | None, max_pages: int | None) -> PdfDocument | ImageDocument:
    """Decode Base64 `file`; fileType 0 is PDF, 1 is image, None detects by magic bytes.

    Raises DocumentError for input the gateway cannot accept.
    """
    if file_type not in (PDF_FILE_TYPE, IMAGE_FILE_TYPE, None):
        raise DocumentError(422, f"Invalid fileType {file_type!r}; expected 0 (PDF), 1 (image) or null")
    try:
        data = base64.b64decode(file, validate=True)
    except binascii.Error:
        raise DocumentError(422, "file is not valid Base64") from None
    if not data:
        raise DocumentError(422, "file is empty after Base64 decoding")

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
