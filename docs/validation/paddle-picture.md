# Picture output verification — issue #7

## Real Paddle sample

The synthetic landscape in [paddle-picture/input.png](paddle-picture/input.png) was processed by the real PaddleOCR-VL pipeline, not fake vLLM or a local Markdown formatter. It uses no private document content.

Reproduce from the repository root in the separate Paddle environment, with a free GPU and a new output directory (existing output directories are refused):

```sh
CUDA_VISIBLE_DEVICES=2 \
  ../legal-vi-ocr/experiments/paddleocr-vl/.venv/bin/python \
  scripts/check_paddle_format.py \
  --model ../legal-vi-ocr/models/PaddleOCR-VL-1.6 \
  --layout ../legal-vi-ocr/models/PP-DocLayoutV3_safetensors \
  --output docs/validation/paddle-picture --picture
```

The run completed on an NVIDIA A100 using the package versions recorded in [paddle-picture/versions.json](paddle-picture/versions.json). The script disables model downloads. Gateway dependencies are unchanged.

Saved evidence:

- `input.png`: synthetic input.
- `response.json`: actual Paddle `result.json` (pipeline output, not HTTP envelope).
- `markdown.md`: actual Paddle `result.markdown["markdown_texts"]`.
- `imgs/*.jpg`: actual Paddle Markdown crop.
- `versions.json`: exact package versions.

The detected image bbox is [298, 1250, 1300, 1802] on a 1600-pixel-wide page. Paddle emitted:

```html
<div style="text-align: center;"><img src="imgs/img_in_image_box_298_1250_1300_1802.jpg" alt="Image" width="62%" /></div>
```

This confirms the centered HTML wrapper and integer width percentage (`int(1002 / 1600 * 100) == 62`) used by the gateway. This is real pipeline evidence for the Markdown wrapper, not an HTTP serving capture.

## Gateway path contract correction

Issues #1 and #7 originally specified bbox-only image paths. That conflicts with the acceptance target: WeKnora merges all pages' image maps and keeps the first value for a duplicate path (`references/WeKnora/internal/infrastructure/docparser/paddleocr_vl_converter.go`, `callLayoutParsing`). Different pictures at matching coordinates would display the first page's picture.

The gateway therefore uses this corrected relative-path contract for every page, including single-page inputs:

```text
imgs/page_{one_based_page_number}/img_in_image_box_{x1}_{y1}_{x2}_{y2}.jpg
```

The Paddle-style filename is unchanged; the page directory makes keys unique within a document. Markdown references exactly the corresponding key. This intentionally supersedes the literal bbox-only path in issues #1/#7; the upstream Paddle sample above remains unchanged as evidence. No issue tracker edits are made by this local fix.

## Resource and regression evidence

Picture output crops are created and JPEG-encoded in the response builder, off the event loop, before the page concurrency slot is released. Parsed blocks no longer retain raw output crops across the document. Disabled or ignored Pictures create no output crops. Two-stage inference still uses pinned Upstream task preparation, including its transient crops.

`tests/test_picture_pages.py` sends three-page PDFs and TIFFs with different pictures at identical coordinates through HTTP and fake vLLM, in both parsing modes. It checks distinct references, decoded image colors/sizes, WeKnora's first-value-wins merge, disabled/ignored images, and crop release. Weak references supplement HTTP checks because raw-image lifetime is not observable in JSON. Reproduce with:

```sh
uv run pytest tests/test_picture_pages.py tests/test_paddle_format.py -q
```
