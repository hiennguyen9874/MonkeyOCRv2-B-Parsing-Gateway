"""Async client for the OpenAI-compatible chat API served by vLLM."""

from __future__ import annotations

import asyncio

import httpx
from PIL import Image

from .config import Settings
from .upstream import image_to_png_data_uri, load_image


def _request_image_uri(image: Image.Image, max_pixels: int, min_pixels: int | None) -> str:
    return image_to_png_data_uri(load_image(image, max_pixels=max_pixels, min_pixels=min_pixels))


class VLLMClient:
    def __init__(self, settings: Settings):
        base_url = settings.vllm_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        self._http = httpx.AsyncClient(base_url=base_url, timeout=settings.request_timeout)
        # vLLM serves /health at the root, outside /v1.
        self._health_url = base_url.removesuffix("/v1") + "/health"
        self._health_check_timeout = settings.health_check_timeout
        self._model = settings.served_model_name
        self._max_pixels = settings.max_pixels

    async def check_ready(self) -> str | None:
        """None when vLLM /health is OK and /v1/models lists the served model, else the reason."""
        try:
            async with asyncio.timeout(self._health_check_timeout):
                health = await self._http.get(self._health_url)
                if health.status_code != 200:
                    return f"vLLM server not ready: /health returned HTTP {health.status_code}"
                models = await self._http.get("/models")
                models.raise_for_status()
                served = [model["id"] for model in models.json()["data"]]
        except TimeoutError:
            return f"vLLM health check timed out after {self._health_check_timeout}s"
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            return f"vLLM server not reachable: {type(exc).__name__}: {exc}"
        if self._model not in served:
            return (
                f"Served model {self._model!r} (MOCR2_SERVED_MODEL_NAME) not found in "
                f"vLLM /v1/models: {served}"
            )
        return None

    async def chat(
        self,
        image: Image.Image,
        prompt: str,
        *,
        min_pixels: int | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send one image + prompt, mirroring MonkeyOCRv2_ServerParsing._chat_completion."""
        image_uri = await asyncio.to_thread(_request_image_uri, image, self._max_pixels, min_pixels)
        payload = {
            "model": self._model,
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        response = await self._http.post("/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    async def aclose(self) -> None:
        await self._http.aclose()
