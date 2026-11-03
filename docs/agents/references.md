# References map

Where to find the behaviour the gateway must match or serve. Paths are relative to the repo root unless they are absolute.

## Upstream: `references/MonkeyOCRv2` (fork, pinned `106914e`)

The authority for inference behaviour. The fork only fixes vLLM 0.11 patch-version detection.

- `parsing/core_runner.py`:
  - `ALL_PROMPT`: prompts per Nhãn Monkey, plus LAYOUT and END2END.
  - `MonkeyOCRv2_ServerParsing`: OpenAI chat payload, retries, and request-image resizing through `batch_inference` → `load_image`.
  - `get_layout`: min_pixels 1,003,520, 4,096 tokens.
  - `parse_end2end_output`, `_parse_tolerant_items`, `_map_bbox_to_image`: model output → bbox in pixels.
  - `_build_page_tasks` and `_recognize_one_block`: Khối crops come from the full-resolution page; 5,000 tokens.
  - `_format_block_fields`, `otsl_to_html`, `process_formula`, `result2md`.
  - The per-page flow for both modes lives in `ServicePipelinePool` (look for the `end2end` branch).
  - The module imports torch and the preprocessor at the top. Port from it rather than importing it.
- `parsing/tests/`: source cases for the parity tests (output parsing, OTSL, bbox, fake server pipeline).
- `parsing/serve.py`: starts vLLM. It chooses the modeling module by vLLM version: `==0.11` or `>=0.25`. The served model name defaults to `MonkeyOCRv2`.
- `parsing/fastapi/main.py`: Upstream's own API (ZIP output). Use it only as a reference for env settings and concurrency.

## PaddleOCR: `references/PADDLEOCR`

- `deploy/paddleocr_vl_docker/hps/gateway/app.py`: HPS gateway. Envelope, `logId`, errorCode → HTTP status, health and readiness, concurrency.
- `docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md`: `prunedResult` fields (`parsing_res_list`, `model_settings`) and the serving schema.
- `docs/paddleocr-api.md` (this repo) summarises both with line links.

## WeKnora client: `references/WeKnora`

`docs/weknora-call-api.md` was written against snapshot `fd7a2586fd58`, and its links use a `raw/WeKnora/` prefix. The submodule is pinned to `02e30e9f5`, and the code is under `references/WeKnora/internal/infrastructure/docparser/`:

- `paddleocr_vl_converter.go`: request payload (`paddleOCRVLRecognitionParams`), response structs, the ping (`GET /layout-parsing`, healthy when < 500), and the 1,000 s timeout.
- `paddleocr_vl_layout.go`: fields read from `prunedResult` (`width`, `height`, `parsing_res_list[].block_content/block_bbox`).
- `mineru_converter.go` (`mineruImageOriginalRefs`): matches `markdown.images` keys against `![](…)` and `<img src>` references.
- `html_table_normalizer.go`: post-processing applied to HTML tables in Markdown.
- Deployment gotcha: WeKnora's `ValidateURLForSSRF` (`internal/utils/security.go`) rejects direct-IP and internal hostnames unless they are listed in `SSRF_WHITELIST`. The gateway endpoint must use an allowed hostname or be whitelisted.

## legal-vi-ocr (sibling repo, absolute paths)

The same model and code were validated here, including the two-stage run (40/40 pages).

- `/home/jovyan/workspace/ocr/legal-vi-ocr/docs/monkeyocr-v2-two-stage.md`: settings, limitations, results.
- `/home/jovyan/workspace/ocr/legal-vi-ocr/src/legal_vi_ocr/monkey_two_stage_run.py`: calls Upstream `get_layout` → `_build_page_tasks` → `_recognize_one_block` in sequence.
- `/home/jovyan/workspace/ocr/legal-vi-ocr/src/legal_vi_ocr/paddle_monkey.py`: `LABELS` maps Nhãn Paddle → Nhãn Monkey, the reverse of the mapping the gateway needs.
- `/home/jovyan/workspace/ocr/legal-vi-ocr/outputs/runs/2026-10-29/paddleocr-vl/`: real PaddleOCR-VL `prunedResult` (`*.raw.json`). Use it to settle format questions; the content is sensitive.
