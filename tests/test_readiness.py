"""GET /health/ready: 200 only when vLLM is healthy and serves the configured model."""

from __future__ import annotations

import socket
import time

from fastapi.testclient import TestClient

from monkeyocr_gateway.app import create_app


def _assert_not_ready(response, *words):
    assert response.status_code == 503
    body = response.json()
    assert body["errorCode"] == 503
    assert isinstance(body["logId"], str) and body["logId"]
    assert "inference server" in body["errorMsg"].lower()
    assert "vllm" not in body["errorMsg"].lower()
    for word in words:
        assert word in body["errorMsg"]


def test_ready_when_vllm_is_healthy_and_serves_the_model(gateway, fake_vllm):
    response = gateway.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["errorCode"] == 0
    assert body["errorMsg"] == "Ready"
    assert isinstance(body["logId"], str) and body["logId"]
    assert fake_vllm.get_paths == ["/health", "/v1/models"]


def test_server_url_with_v1_suffix_still_checks_root_health(fake_vllm, monkeypatch):
    monkeypatch.setenv("MOCR2_SERVER_URL", fake_vllm.url + "/v1/")
    with TestClient(create_app()) as client:
        assert client.get("/health/ready").status_code == 200
    assert fake_vllm.get_paths == ["/health", "/v1/models"]


def test_not_ready_when_vllm_health_fails(gateway, fake_vllm):
    fake_vllm.health_status = 500

    _assert_not_ready(gateway.get("/health/ready"), "/health", "HTTP 500")
    assert fake_vllm.get_paths == ["/health"]


def test_not_ready_when_configured_model_is_not_served(gateway, fake_vllm):
    fake_vllm.served_models = ["other-model"]

    _assert_not_ready(gateway.get("/health/ready"), "fake-monkey", "MOCR2_SERVED_MODEL_NAME", "/v1/models")


def test_not_ready_when_vllm_is_unreachable(monkeypatch):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    # Nothing listens on the port once the socket is closed.
    monkeypatch.setenv("MOCR2_SERVER_URL", f"http://127.0.0.1:{port}")
    with TestClient(create_app()) as client:
        _assert_not_ready(client.get("/health/ready"), "not reachable")


def test_not_ready_when_vllm_times_out_within_the_configured_timeout(fake_vllm, monkeypatch):
    monkeypatch.setenv("MOCR2_HEALTH_CHECK_TIMEOUT", "0.2")
    fake_vllm.get_delay = 1.0
    with TestClient(create_app()) as client:
        start = time.monotonic()
        response = client.get("/health/ready")
        elapsed = time.monotonic() - start

    _assert_not_ready(response, "timed out")
    assert elapsed < 0.9
