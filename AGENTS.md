# MonkeyOCRv2-B-Parsing Gateway

A FastAPI gateway that serves the PaddleOCR-VL `/layout-parsing` contract in front of a vLLM server running MonkeyOCRv2-B-Parsing. PaddleOCR-VL clients, WeKnora first, can switch engines without code changes. It supports both two-stage and end-to-end parsing; see `GLOSSARY.md`.

## Work

- The spec is GitHub issue #1. The tickets are #2–#11 and use native blocked-by links. Take a ticket from the frontier: one whose blockers are all closed.
- A ticket is done when every acceptance criterion is checked, with evidence from the test seams below.

## Contract

- WeKnora is the acceptance target. `docs/weknora-call-api.md` lists what it sends and what it reads. Read it before changing request handling or the response shape.
- `docs/paddleocr-api.md` describes the PaddleOCR-VL HTTP schema.

## Invariants

- Upstream (`references/MonkeyOCRv2` at `106914e`) is the authority for prompts, output parsing, cropping, and token/pixel limits. Port its helpers with the source revision noted and add a parity test. The gateway itself stays free of torch and GPUs (ADR-0003).
- Tests use two seams: (1) the gateway HTTP API running in-process against a fake OpenAI-compatible vLLM that replays recorded Monkey output, and (2) parity tests on the ported helpers. Real GPU runs belong only to the manual smoke script.
- Outputs from the sibling `legal-vi-ocr` repo contain sensitive documents. Read them to learn the shape. Build fixtures here only from synthetic or public material.

## Setup

- `git submodule update --init` checks out `references/` at the pinned revisions.
- `uv sync` installs the gateway (Python 3.12). `uv run pytest` runs both test seams. `uv run uvicorn --factory monkeyocr_gateway.app:create_app` starts the gateway; `MOCR2_SERVER_URL` and `MOCR2_SERVED_MODEL_NAME` point it at vLLM; `MOCR2_PARSING_MODE` (`two-stage` by default, or `end-to-end`) sets the mode for requests without `vlmExtraArgs.parsingMode`. `MOCR2_PAGE_CONCURRENCY` (positive integer, default 8) caps pages parsed in parallel per request; `MOCR2_MAX_BODY_BYTES` (default 100 MiB) and `MOCR2_MAX_PAGES` (default 0, unlimited) bound the input.
- `models/` is gitignored. Download the Hugging Face repo `zenosai/MonkeyOCRv2-B-Parsing` into `models/MonkeyOCRv2-B-Parsing`. Upstream's `download_model.py` writes to its own `model_weight/` directory instead.

## Docs

- `docs/agents/references.md`: read it when you need upstream behaviour, PaddleOCR schema details, WeKnora client code, or the legal-vi-ocr two-stage runner.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default five triage labels. See `docs/agents/triage-labels.md`.

### Domain docs

This repo uses a single-context layout. See `docs/agents/domain.md`.
