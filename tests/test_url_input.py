"""`file` as a URL: downloaded only from hosts in MOCR2_URL_ALLOWLIST (SSRF guard)."""

from __future__ import annotations

import base64
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import image_base64
from fastapi.testclient import TestClient
from PIL import Image
from test_multi_page import (  # noqa: F401 - fixture
    _contents,
    _pages,
    _pdf_base64,
    _tiff_base64,
    fake_vllm,
)

from monkeyocr_gateway.app import create_app


class FileServer:
    """Serves routes {path: (status, headers, body)}; records every path requested."""

    def __init__(self):
        self.routes: dict[str, tuple[int, dict, bytes]] = {}
        self.paths: list[str] = []
        self.delay = 0.0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - stdlib handler API
                server.paths.append(self.path)
                time.sleep(server.delay)
                status, headers, body = server.routes.get(self.path, (404, {}, b""))
                try:
                    self.send_response(status)
                    for name, value in headers.items():
                        self.send_header(name, value)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass  # the gateway gave up (timeout and size tests)

            def log_message(self, *_args):
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def url(self, path: str, host: str = "127.0.0.1") -> str:
        return f"http://{host}:{self._server.server_port}{path}"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def files(fake_vllm, monkeypatch):
    for name in ("MOCR2_URL_ALLOWLIST", "MOCR2_URL_FETCH_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    with FileServer() as server:
        yield server


def _post(client, file: str, file_type=None, **extra):
    return client.post(
        "/layout-parsing",
        json={"file": file, "fileType": file_type, "vlmExtraArgs": {"parsingMode": "end-to-end"}, **extra},
    )


def _error(response, status: int, *words: str) -> dict:
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"logId", "errorCode", "errorMsg"}
    assert body["errorCode"] == status
    for word in words:
        assert word in body["errorMsg"]
    return body


def _serve(files, path: str, data_base64: str):
    files.routes[path] = (200, {}, base64.b64decode(data_base64))


def test_urls_are_rejected_when_the_allowlist_is_empty(files, fake_vllm):
    _serve(files, "/page.png", image_base64(_pages(1)[0]))
    with TestClient(create_app()) as client:
        body = _error(_post(client, files.url("/page.png"), logId="client-log"), 422, "MOCR2_URL_ALLOWLIST")

    assert body["logId"] == "client-log"
    assert files.paths == []
    assert fake_vllm.requests == []


@pytest.mark.parametrize(
    ("path", "encode", "file_type", "pages"),
    [
        ("/page.png", lambda: image_base64(_pages(1)[0]), None, 1),
        ("/doc.pdf", lambda: _pdf_base64(_pages(3)), None, 3),
        ("/doc.pdf", lambda: _pdf_base64(_pages(2)), 0, 2),
        ("/scan.tiff", lambda: _tiff_base64(_pages(2)), 1, 2),
    ],
)
def test_allowlisted_url_is_parsed_like_base64(files, monkeypatch, path, encode, file_type, pages):
    monkeypatch.setenv("MOCR2_URL_ALLOWLIST", "example.invalid, 127.0.0.1")
    data = encode()
    _serve(files, path, data)
    with TestClient(create_app()) as client:
        from_url = _post(client, files.url(path), file_type)
        from_base64 = _post(client, data, file_type)

    assert from_url.status_code == 200, from_url.text
    assert from_url.json()["result"] == from_base64.json()["result"]
    assert _contents(from_url.json()["result"]) == [f"Trang {n}" for n in range(1, pages + 1)]


def test_host_match_is_case_insensitive(files, monkeypatch):
    monkeypatch.setenv("MOCR2_URL_ALLOWLIST", "LocalHost")
    _serve(files, "/page.png", image_base64(_pages(1)[0]))
    with TestClient(create_app()) as client:
        assert _post(client, files.url("/page.png", host="LOCALHOST")).status_code == 200


def test_host_outside_the_allowlist_is_rejected(files, monkeypatch, fake_vllm):
    monkeypatch.setenv("MOCR2_URL_ALLOWLIST", "127.0.0.1")
    _serve(files, "/page.png", image_base64(_pages(1)[0]))
    with TestClient(create_app()) as client:
        _error(_post(client, files.url("/page.png", host="localhost")), 422, "'localhost'", "MOCR2_URL_ALLOWLIST")

    assert files.paths == []
    assert fake_vllm.requests == []


def test_redirect_within_the_allowlist_is_followed(files, monkeypatch):
    monkeypatch.setenv("MOCR2_URL_ALLOWLIST", "127.0.0.1")
    _serve(files, "/page.png", image_base64(_pages(1)[0]))
    files.routes["/moved"] = (302, {"Location": "/page.png"}, b"")
    with TestClient(create_app()) as client:
        assert _contents(_post(client, files.url("/moved")).json()["result"]) == ["Trang 1"]

    assert files.paths == ["/moved", "/page.png"]


def test_redirect_to_a_host_outside_the_allowlist_is_rejected(files, monkeypatch, fake_vllm):
    monkeypatch.setenv("MOCR2_URL_ALLOWLIST", "127.0.0.1")
    _serve(files, "/page.png", image_base64(_pages(1)[0]))
    files.routes["/moved"] = (302, {"Location": files.url("/page.png", host="localhost")}, b"")
    with TestClient(create_app()) as client:
        _error(_post(client, files.url("/moved")), 422, "'localhost'", "MOCR2_URL_ALLOWLIST")

    assert files.paths == ["/moved"]
    assert fake_vllm.requests == []


def test_redirect_loop_is_rejected(files, monkeypatch):
    monkeypatch.setenv("MOCR2_URL_ALLOWLIST", "127.0.0.1")
    files.routes["/loop"] = (302, {"Location": "/loop"}, b"")
    with TestClient(create_app()) as client:
        _error(_post(client, files.url("/loop")), 422, "redirected more than 5 times")

    assert len(files.paths) == 6


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://127.0.0.1/page.png"])
def test_non_http_schemes_are_rejected(files, monkeypatch, url):
    monkeypatch.setenv("MOCR2_URL_ALLOWLIST", "127.0.0.1")
    with TestClient(create_app()) as client:
        _error(_post(client, url), 422, "scheme")


def test_download_over_the_size_limit_is_413(files, monkeypatch, fake_vllm):
    monkeypatch.setenv("MOCR2_URL_ALLOWLIST", "127.0.0.1")
    monkeypatch.setenv("MOCR2_MAX_BODY_BYTES", "2000")
    _serve(files, "/large.png", image_base64(Image.effect_noise((100, 100), 64)))
    _serve(files, "/small.png", image_base64(Image.new("RGB", (10, 10))))
    with TestClient(create_app()) as client:
        _error(_post(client, files.url("/large.png")), 413, "2000 bytes", "MOCR2_MAX_BODY_BYTES")
        assert _post(client, files.url("/small.png")).status_code == 200


def test_download_over_the_time_limit_is_504(files, monkeypatch, fake_vllm):
    monkeypatch.setenv("MOCR2_URL_ALLOWLIST", "127.0.0.1")
    monkeypatch.setenv("MOCR2_URL_FETCH_TIMEOUT", "0.3")
    _serve(files, "/slow.png", image_base64(_pages(1)[0]))
    files.delay = 1.0
    with TestClient(create_app()) as client:
        _error(_post(client, files.url("/slow.png")), 504, "0.3s", "MOCR2_URL_FETCH_TIMEOUT")

    assert fake_vllm.requests == []


def test_failed_download_is_a_clear_error(files, monkeypatch):
    monkeypatch.setenv("MOCR2_URL_ALLOWLIST", "127.0.0.1")
    with TestClient(create_app()) as client:
        _error(_post(client, files.url("/missing.png")), 422, "HTTP 404")
