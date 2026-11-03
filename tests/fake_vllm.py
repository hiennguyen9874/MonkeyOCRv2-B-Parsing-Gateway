"""Fake OpenAI-compatible vLLM server that replays recorded MonkeyOCRv2 output.

Answers are chosen by the prompt text, so a test controls what the "model"
returns for each Upstream prompt and can inspect every payload it received.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeVLLM:
    def __init__(self, answers: dict[str, str]):
        self.answers = answers
        self.requests: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - stdlib handler API
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append({"path": self.path, "payload": payload})
                prompt = next(
                    item["text"]
                    for item in payload["messages"][0]["content"]
                    if item["type"] == "text"
                )
                body = json.dumps({
                    "choices": [{
                        "message": {"role": "assistant", "content": fake.answers[prompt]},
                        "finish_reason": "stop",
                    }],
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

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
