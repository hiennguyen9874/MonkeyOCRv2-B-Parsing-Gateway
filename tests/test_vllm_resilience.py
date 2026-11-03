"""Resilience and load control on vLLM calls: retries with backoff, error mapping,
timeouts, truncated output, opt-in repeat retry and concurrency limits."""

from __future__ import annotations

import json
import socket
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import END2END_PROMPT, LAYOUT_PROMPT, TEXT_PROMPT, image_base64
from fake_vllm import FakeVLLM, request_prompt
from monkeyocr_gateway.app import create_app

ENV = (
    "MOCR2_HTTP_MAX_RETRIES", "MOCR2_HTTP_RETRY_BACKOFF", "MOCR2_REQUEST_TIMEOUT",
    "MOCR2_RETRY_REPEAT", "MOCR2_REC_MAX_RETRIES", "MOCR2_REQUEST_CONCURRENCY",
    "MOCR2_SERVER_MAX_INFLIGHT", "MOCR2_PAGE_CONCURRENCY", "MOCR2_PARSING_MODE",
)
LAYOUT_ONE_TEXT = json.dumps([{"bbox": [0, 0, 1000, 1000], "label": "Text"}])
TEXT = "Điều 1. Văn bản tổng hợp."
END2END_ONE_TEXT = json.dumps([{"bbox": [0, 0, 1000, 1000], "label": "Text", "content": TEXT}])
# Output that Upstream's detector flags as repeated tokens.
REPEATED = "Điều " + "ha" * 300
PAGE = image_base64(Image.new("RGB", (200, 200), "white"))
# The prompt that reads Khối content in each mode.
CONTENT_PROMPT = {"two-stage": TEXT_PROMPT, "end-to-end": END2END_PROMPT}


@pytest.fixture
def fake_vllm(monkeypatch):
    answers = {LAYOUT_PROMPT: LAYOUT_ONE_TEXT, TEXT_PROMPT: TEXT, END2END_PROMPT: END2END_ONE_TEXT}
    with FakeVLLM(answers) as fake:
        monkeypatch.setenv("MOCR2_SERVER_URL", fake.url)
        monkeypatch.setenv("MOCR2_SERVED_MODEL_NAME", "fake-monkey")
        for name in ENV:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("MOCR2_HTTP_RETRY_BACKOFF", "0")
        yield fake


def _post(client, mode="two-stage", file=PAGE, **extra):
    return client.post(
        "/layout-parsing",
        json={"file": file, "fileType": None, "vlmExtraArgs": {"parsingMode": mode}, **extra},
    )


def _contents(response) -> list[str]:
    assert response.status_code == 200, response.text
    pages = response.json()["result"]["layoutParsingResults"]
    return [block["block_content"] for page in pages for block in page["prunedResult"]["parsing_res_list"]]


def _error(response, status: int, *words: str) -> dict:
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"logId", "errorCode", "errorMsg"}
    assert body["errorCode"] == status
    for word in words:
        assert word in body["errorMsg"]
    return body


def _calls(fake, prompt: str) -> list[dict]:
    return [r["payload"] for r in fake.requests if request_prompt(r["payload"]) == prompt]


def test_transient_vllm_errors_are_retried_with_exponential_backoff(fake_vllm, monkeypatch):
    monkeypatch.setenv("MOCR2_HTTP_MAX_RETRIES", "3")
    monkeypatch.setenv("MOCR2_HTTP_RETRY_BACKOFF", "0.1")
    fake_vllm.post_statuses = [503, 429, 502]
    with TestClient(create_app()) as client:
        assert _contents(_post(client, "end-to-end")) == [TEXT]

    assert len(fake_vllm.requests) == 4
    gaps = [b - a for a, b in zip(fake_vllm.post_times, fake_vllm.post_times[1:])]
    for gap, backoff in zip(gaps, [0.1, 0.2, 0.4]):
        assert backoff * 0.9 <= gap < backoff + 0.3


@pytest.mark.parametrize("status", [501, 507, 599])
def test_other_5xx_responses_are_retried(fake_vllm, status):
    fake_vllm.post_statuses = [status]
    with TestClient(create_app()) as client:
        assert _contents(_post(client, "end-to-end")) == [TEXT]

    assert len(fake_vllm.requests) == 2


@pytest.mark.parametrize(
    ("mode", "failing_status"), [("two-stage", 503), ("end-to-end", 500), ("end-to-end", 507)]
)
def test_vllm_failing_after_retries_fails_the_whole_request(fake_vllm, monkeypatch, mode, failing_status):
    monkeypatch.setenv("MOCR2_HTTP_MAX_RETRIES", "2")
    fake_vllm.answers[CONTENT_PROMPT[mode]] = failing_status
    with TestClient(create_app()) as client:
        body = _error(_post(client, mode, logId="client-log"), 503, "vLLM", "3 attempts")
        assert body["logId"] == "client-log"
        # The gateway keeps serving once vLLM recovers.
        fake_vllm.answers[CONTENT_PROMPT[mode]] = {"two-stage": TEXT, "end-to-end": END2END_ONE_TEXT}[mode]
        assert _contents(_post(client, mode)) == [TEXT]

    assert len(_calls(fake_vllm, CONTENT_PROMPT[mode])) == 3 + 1


def test_unreachable_vllm_is_503_after_retrying_the_connection(monkeypatch):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    monkeypatch.setenv("MOCR2_SERVER_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("MOCR2_HTTP_MAX_RETRIES", "1")
    monkeypatch.setenv("MOCR2_HTTP_RETRY_BACKOFF", "0")
    with TestClient(create_app()) as client:
        _error(_post(client), 503, "vLLM", "2 attempts", "Connect")


def test_non_retryable_vllm_error_is_500_without_retry(fake_vllm):
    fake_vllm.answers[END2END_PROMPT] = 400
    with TestClient(create_app()) as client:
        _error(_post(client, "end-to-end"), 500, "HTTP 400")

    assert len(fake_vllm.requests) == 1


def test_vllm_timeout_is_504(fake_vllm, monkeypatch):
    monkeypatch.setenv("MOCR2_REQUEST_TIMEOUT", "0.3")
    fake_vllm.post_delay = 1.0
    with TestClient(create_app()) as client:
        _error(_post(client, "end-to-end"), 504, "0.3s", "MOCR2_REQUEST_TIMEOUT")

    # A call that used the whole timeout is not retried.
    assert len(fake_vllm.requests) == 1


def test_vllm_connect_timeout_is_504_without_retry(fake_vllm, monkeypatch):
    attempts = []

    async def connect_timeout(_transport, request):
        attempts.append(request)
        raise httpx.ConnectTimeout("Connection timed out", request=request)

    # A real TCP blackhole is not deterministic; inject the transport failure
    # while exercising the gateway's HTTP error response.
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", connect_timeout)
    with TestClient(create_app()) as client:
        _error(_post(client, "end-to-end", logId="connect-timeout"), 504, "MOCR2_REQUEST_TIMEOUT")

    assert len(attempts) == 1
    assert fake_vllm.requests == []


def test_finish_reason_length_keeps_partial_content(fake_vllm):
    fake_vllm.finish_reason = "length"
    fake_vllm.answers[TEXT_PROMPT] = "Điều 1. Văn bản bị cắt ng"
    with TestClient(create_app()) as client:
        assert _contents(_post(client)) == ["Điều 1. Văn bản bị cắt ng"]


@pytest.mark.parametrize("mode", ["two-stage", "end-to-end"])
def test_repeat_retry_is_off_by_default(fake_vllm, mode):
    fake_vllm.answers[CONTENT_PROMPT[mode]] = REPEATED
    with TestClient(create_app()) as client:
        assert _post(client, mode).status_code == 200

    assert len(_calls(fake_vllm, CONTENT_PROMPT[mode])) == 1


@pytest.mark.parametrize("mode", ["two-stage", "end-to-end"])
def test_repeat_retry_asks_again_with_rising_temperature(fake_vllm, monkeypatch, mode):
    monkeypatch.setenv("MOCR2_RETRY_REPEAT", "1")
    good = {"two-stage": TEXT, "end-to-end": END2END_ONE_TEXT}[mode]
    answers = iter([REPEATED, REPEATED, good])
    fake_vllm.answers[CONTENT_PROMPT[mode]] = lambda _image: next(answers)
    with TestClient(create_app()) as client:
        assert _contents(_post(client, mode, repetitionPenalty=1.05)) == [TEXT]

    calls = _calls(fake_vllm, CONTENT_PROMPT[mode])
    assert [(c["temperature"], c.get("top_p")) for c in calls] == [(0, None), (0.2, 0.95), (0.4, 0.95)]
    # Other generation options are kept on retries.
    assert {c["repetition_penalty"] for c in calls} == {1.05}
    # Layout output is never retried, as in Upstream.
    assert len(_calls(fake_vllm, LAYOUT_PROMPT)) == (1 if mode == "two-stage" else 0)


def test_repeat_retry_stops_at_the_configured_limit(fake_vllm, monkeypatch):
    monkeypatch.setenv("MOCR2_RETRY_REPEAT", "true")
    monkeypatch.setenv("MOCR2_REC_MAX_RETRIES", "5")
    fake_vllm.answers[TEXT_PROMPT] = REPEATED
    with TestClient(create_app()) as client:
        assert _contents(_post(client)) == [REPEATED]

    temperatures = [c["temperature"] for c in _calls(fake_vllm, TEXT_PROMPT)]
    assert temperatures == pytest.approx([0, 0.2, 0.4, 0.6, 0.8, 0.8])


def _slow(answer: str):
    def slow_answer(_image):
        time.sleep(0.2)
        return answer
    return slow_answer


def test_vllm_calls_in_flight_never_exceed_the_process_cap(fake_vllm, monkeypatch):
    monkeypatch.setenv("MOCR2_SERVER_MAX_INFLIGHT", "3")
    fake_vllm.answers[END2END_PROMPT] = _slow(END2END_ONE_TEXT)
    with TestClient(create_app()) as client:
        responses = _run_in_parallel(client, 8, "end-to-end")

    assert all(_contents(response) == [TEXT] for response in responses)
    assert fake_vllm.max_in_flight == 3


@pytest.mark.parametrize(("env", "requests", "expected"), [(None, 18, 16), ("2", 5, 2)])
def test_requests_beyond_the_limit_wait_in_a_queue(fake_vllm, monkeypatch, env, requests, expected):
    if env is not None:
        monkeypatch.setenv("MOCR2_REQUEST_CONCURRENCY", env)
    fake_vllm.answers[END2END_PROMPT] = _slow(END2END_ONE_TEXT)
    with TestClient(create_app()) as client:
        responses = _run_in_parallel(client, requests, "end-to-end")

    assert all(_contents(response) == [TEXT] for response in responses)
    # One end-to-end page is one vLLM call, so calls in flight count requests in progress.
    assert fake_vllm.max_in_flight == expected


def _run_in_parallel(client, count: int, mode: str) -> list:
    responses = [None] * count

    def run(index):
        responses[index] = _post(client, mode)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return responses


@pytest.mark.parametrize("name", ["MOCR2_REQUEST_CONCURRENCY", "MOCR2_SERVER_MAX_INFLIGHT"])
def test_nonpositive_concurrency_limits_are_rejected_at_startup(monkeypatch, name):
    monkeypatch.setenv(name, "0")
    with pytest.raises(ValueError, match=f"{name} must be positive"):
        create_app()
