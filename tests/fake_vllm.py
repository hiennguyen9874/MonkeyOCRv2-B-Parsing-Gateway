"""Fake OpenAI-compatible vLLM server that replays recorded MonkeyOCRv2 output.

Answers are chosen by the prompt text, so a test controls what the "model"
returns for each Upstream prompt and can inspect every payload it received.
An answer may also be a callable taking the request image; it runs on the
server's handler thread, so tests can coordinate out-of-order responses.
An answer that is an int is an HTTP error status returned for that prompt.
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
        for item in next(m["content"] for m in payload["messages"] if m["role"] == "user")
        if item["type"] == "image_url"
    )
    return Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1])))


def request_prompt(payload: dict) -> str:
    return next(
        item["text"] for item in next(m["content"] for m in payload["messages"] if m["role"] == "user")
        if item["type"] == "text"
    )


class FakeVLLM:
    def __init__(self, answers: dict[str, str | int | Callable[[Image.Image], str]]):
        self.answers = answers
        # Chat requests: statuses returned, one per request, before answering normally.
        self.post_statuses: list[int] = []
        self.post_delay = 0.0
        self.reject_negative_repeat_last_n = False
        self.finish_reason = "stop"
        # time.monotonic() of each chat request received.
        self.post_times: list[float] = []
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
                    fake.post_times.append(time.monotonic())
                    fake.in_flight += 1
                    fake.max_in_flight = max(fake.max_in_flight, fake.in_flight)
                    status = fake.post_statuses.pop(0) if fake.post_statuses else 200
                time.sleep(fake.post_delay)
                answer = fake.answers[request_prompt(payload)]
                if fake.reject_negative_repeat_last_n and payload.get("repeat_last_n", 0) < 0:
                    status = 400
                if status == 200 and isinstance(answer, int):
                    status = answer
                if status != 200:
                    body = json.dumps({"object": "error", "message": "fake failure"}).encode()
                else:
                    if callable(answer):
                        answer = answer(request_image(payload))
                    body = json.dumps({
                        "choices": [{
                            "message": {"role": "assistant", "content": answer},
                            "finish_reason": fake.finish_reason,
                        }],
                    }).encode()
                with fake.completed_condition:
                    try:
                        self.send_response(status)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                    except OSError:
                        pass  # the gateway gave up waiting (timeout tests)
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
