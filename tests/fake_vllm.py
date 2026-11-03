"""Fake OpenAI-compatible vLLM server that replays recorded MonkeyOCRv2 output.

Answers are chosen by the prompt text, so a test controls what the "model"
returns for each Upstream prompt and can inspect every payload it received.
An answer may also be a callable taking the request image; it runs on the
server's handler thread, so tests can coordinate out-of-order responses.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

from PIL import Image


def request_image(payload: dict) -> Image.Image:
    """Decode the PNG data URI sent in a chat payload."""
    url = next(
        item["image_url"]["url"]
        for item in payload["messages"][0]["content"]
        if item["type"] == "image_url"
    )
    return Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1])))


def request_prompt(payload: dict) -> str:
    return next(
        item["text"] for item in payload["messages"][0]["content"] if item["type"] == "text"
    )


class FakeVLLM:
    def __init__(self, answers: dict[str, str | Callable[[Image.Image], str]]):
        self.answers = answers
        # Readiness endpoints: tests change these to simulate an unhealthy server.
        self.health_status = 200
        self.served_models = ["fake-monkey"]
        self.get_delay = 0.0
        self.get_paths: list[str] = []
        self.requests: list[dict] = []
        # Payloads in the order their responses were sent.
        self.completed: list[dict] = []
        self.completed_condition = threading.Condition()
        # Requests received but not yet answered, and the highest such count seen.
        self.in_flight = 0
        self.max_in_flight = 0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - stdlib handler API
                fake.get_paths.append(self.path)
                time.sleep(fake.get_delay)
                if self.path == "/health":
                    status, body = fake.health_status, b""
                elif self.path == "/v1/models":
                    status = 200
                    body = json.dumps({
                        "object": "list",
                        "data": [{"id": name, "object": "model"} for name in fake.served_models],
                    }).encode()
                else:
                    status, body = 404, b""
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # noqa: N802 - stdlib handler API
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append({"path": self.path, "payload": payload})
                with fake.completed_condition:
                    fake.in_flight += 1
                    fake.max_in_flight = max(fake.max_in_flight, fake.in_flight)
                answer = fake.answers[request_prompt(payload)]
                if callable(answer):
                    answer = answer(request_image(payload))
                body = json.dumps({
                    "choices": [{
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }],
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                with fake.completed_condition:
                    self.wfile.write(body)
                    fake.in_flight -= 1
                    fake.completed.append(payload)
                    fake.completed_condition.notify_all()

            def log_message(self, *_args):
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
