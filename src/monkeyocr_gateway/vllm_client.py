"""Async client for the OpenAI-compatible chat API served by vLLM."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx
from PIL import Image

from .config import Settings
from .upstream import image_to_png_data_uri, load_image

logger = logging.getLogger(__name__)

MAX_RETRY_SLEEP = 30.0


class VLLMError(Exception):
    """A vLLM call that failed for good; status_code is the HTTP status to return:
    503 vLLM unavailable after retries, 504 timeout, 500 anything else."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def _request_image_uri(image: Image.Image, max_pixels: int, min_pixels: int | None) -> str:
    return image_to_png_data_uri(load_image(image, max_pixels=max_pixels, min_pixels=min_pixels))


@dataclass(frozen=True)
class GenerationOptions:
    """Request generation options; None keeps the Upstream default."""

    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    # Overrides the per-stage max_tokens default when set.
    max_new_tokens: int | None = None


class VLLMClient:
    def __init__(self, settings: Settings):
        base_url = settings.vllm_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        # The in-flight semaphore bounds connections, so the pool itself is unbounded.
        self._http = httpx.AsyncClient(
            base_url=base_url,
            timeout=settings.request_timeout,
            limits=httpx.Limits(max_connections=None),
        )
        self._timeout = settings.request_timeout
        self._max_retries = settings.http_max_retries
        self._retry_backoff = settings.http_retry_backoff
        self._inflight = asyncio.Semaphore(settings.vllm_max_inflight)
        # vLLM serves /health at the root, outside /v1.
        self._health_url = base_url.removesuffix("/v1") + "/health"
        self._health_check_timeout = settings.health_check_timeout
        self._model = settings.served_model_name
        self._backend = settings.server_backend
        self._max_pixels = settings.max_pixels

    async def check_ready(self) -> str | None:
        """None when vLLM /health is OK and /v1/models lists the served model, else the reason."""
        try:
            async with asyncio.timeout(self._health_check_timeout):
                health = await self._http.get(self._health_url)
                if health.status_code != 200:
                    return f"Inference server not ready: /health returned HTTP {health.status_code}"
                models = await self._http.get("/models")
                models.raise_for_status()
                served = [model["id"] for model in models.json()["data"]]
        except TimeoutError:
            return f"Inference server health check timed out after {self._health_check_timeout}s"
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            return f"Inference server not reachable: {type(exc).__name__}: {exc}"
        if self._model not in served:
            return (
                f"Served model {self._model!r} (MOCR2_SERVED_MODEL_NAME) not found in "
                f"inference server /v1/models: {served}"
            )
        return None

    async def chat(
        self,
        image: Image.Image,
        prompt: str,
        options: GenerationOptions,
        *,
        min_pixels: int | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send one image + prompt, mirroring MonkeyOCRv2_ServerParsing._chat_completion.

        max_tokens is the stage default, replaced by options.max_new_tokens when set.
        Raises VLLMError when the call fails after retries. Output cut at the token
        limit (finish reason length) is returned as is.
        """
        image_uri = await asyncio.to_thread(_request_image_uri, image, self._max_pixels, min_pixels)
        payload = {
            "model": self._model,
            "temperature": 0 if options.temperature is None else options.temperature,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        if self._backend == "llama-cpp":
            payload["messages"].insert(0, {"role": "system", "content": "You are a helpful assistant."})
        if options.max_new_tokens is not None:
            max_tokens = options.max_new_tokens
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if options.top_p is not None:
            payload["top_p"] = options.top_p
        if options.repetition_penalty is not None:
            if self._backend == "llama-cpp":
                payload["repeat_penalty"] = options.repetition_penalty
                payload["repeat_last_n"] = -1
            else:
                payload["repetition_penalty"] = options.repetition_penalty
        response = await self._post_with_retries(payload)
        if response.is_error:
            raise VLLMError(500, f"Inference server returned HTTP {response.status_code}: {response.text[:500]}")
        try:
            choice = response.json()["choices"][0]
            content = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise VLLMError(500, f"Inference server returned a malformed chat completion: {exc!r}") from None
        if choice.get("finish_reason") == "length":
            logger.warning("Inference server output hit max_tokens (finish reason length); keeping partial content")
        return content

    async def _post_with_retries(self, payload: dict) -> httpx.Response:
        """Retry 429/5xx and connection errors with exponential backoff, as Upstream does.

        Unlike Upstream, a timeout is not retried: one call already used the
        whole MOCR2_REQUEST_TIMEOUT, and retrying would outlast the client's own timeout.
        """
        last_error = ""
        for attempt in range(self._max_retries + 1):
            if attempt:
                await asyncio.sleep(min(self._retry_backoff * 2 ** (attempt - 1), MAX_RETRY_SLEEP))
            try:
                # The slot is held per attempt, not while sleeping between attempts.
                async with self._inflight:
                    response = await self._http.post("/chat/completions", json=payload)
            except httpx.TimeoutException:
                raise VLLMError(504, f"Inference server did not answer within {self._timeout}s (MOCR2_REQUEST_TIMEOUT)") from None
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            if response.status_code != 429 and not response.is_server_error:
                return response
            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
        raise VLLMError(
            503, f"Inference server unavailable after {self._max_retries + 1} attempts "
            f"(MOCR2_HTTP_MAX_RETRIES): {last_error}"
        )

    async def aclose(self) -> None:
        await self._http.aclose()
