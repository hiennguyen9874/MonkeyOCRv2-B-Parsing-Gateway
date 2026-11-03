"""Cancellation checks that cannot be observed from the fake HTTP server."""

from __future__ import annotations

import asyncio

import pytest
from PIL import Image

from conftest import LAYOUT_PROMPT
from monkeyocr_gateway.page_parser import parse_page


@pytest.mark.parametrize("failure", ["recognition", "request_cancelled"])
def test_two_stage_awaits_sibling_cancellation_before_returning(failure):
    async def run():
        sibling_started = asyncio.Event()
        sibling_cancelled = asyncio.Event()
        error = RuntimeError("recognition failed")

        class Client:
            async def chat(self, image, prompt, **kwargs):
                if prompt == LAYOUT_PROMPT:
                    return '[{"bbox": [0, 0, 500, 500], "label": "Text"}, {"bbox": [500, 500, 1000, 1000], "label": "Title"}]'
                if image.getpixel((0, 0)) == (0, 0, 0):
                    await sibling_started.wait()
                    if failure == "recognition":
                        raise error
                    await asyncio.Event().wait()
                else:
                    sibling_started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        sibling_cancelled.set()

        image = Image.new("RGB", (100, 100), "white")
        image.paste((0, 0, 0), (0, 0, 50, 50))
        parsing = asyncio.create_task(parse_page(image, "two-stage", Client()))
        try:
            await asyncio.wait_for(sibling_started.wait(), timeout=2)
            if failure == "request_cancelled":
                parsing.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await parsing
            else:
                with pytest.raises(RuntimeError) as caught:
                    await parsing
                assert caught.value is error
            assert sibling_cancelled.is_set()
        finally:
            # Clean up even when the regression assertion fails.
            parsing.cancel()
            await asyncio.gather(parsing, return_exceptions=True)
            pending = asyncio.all_tasks() - {asyncio.current_task()}
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(run())
