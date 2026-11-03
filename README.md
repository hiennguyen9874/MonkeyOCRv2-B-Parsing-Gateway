# MonkeyOCRv2-B-Parsing Gateway

A FastAPI gateway that exposes the PaddleOCR-VL `/layout-parsing` API over a vLLM server running MonkeyOCRv2-B-Parsing. It is intended for PaddleOCR-VL clients such as WeKnora.

## Run locally with uv

Run the commands below from the repository root. You need Python 3.12, `uv`, an NVIDIA GPU and a compatible driver. The vLLM installation is large.

Initialize the pinned upstream checkout and download the model if needed:

```sh
git submodule update --init
uv run --with huggingface-hub hf download zenosai/MonkeyOCRv2-B-Parsing \
  --local-dir models/MonkeyOCRv2-B-Parsing
```

Skip the download if the model is already present. The model directory is gitignored. Use trusted model files and upstream code: the server loads model code.

Choose **one** CUDA variant and use the same extra in all `uv run` commands below. The two extras cannot be installed together.

```sh
uv sync --locked --extra vllm-cu126
# Alternatively: uv sync --locked --extra vllm-cu128
```

### 1. Start vLLM

In the first terminal, choose an unused GPU (GPU 0 here). Use the upstream launcher, which registers MonkeyOCR's custom vLLM model implementation:

```sh
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
  uv run --locked --extra vllm-cu126 python references/MonkeyOCRv2/parsing/serve.py \
  -m "$PWD/models/MonkeyOCRv2-B-Parsing" \
  --host 127.0.0.1 -p 19388 --served-model-name MonkeyOCRv2 \
  --gpu-memory-utilization 0.4 --max-model-len 16384 \
  --max-num-batched-tokens 16384 --max-num-seqs 1
```

For CUDA 12.8, replace `vllm-cu126` with `vllm-cu128`. These server settings follow the two-stage experiment in `legal-vi-ocr`; adjust GPU selection and memory utilization for your machine. Keep the server on loopback unless access is appropriately secured.

Wait for the model to load, then check that `MonkeyOCRv2` is listed:

```sh
curl -fsS http://127.0.0.1:19388/v1/models
```

### 2. Start the gateway

In a second terminal, from the repository root:

```sh
MOCR2_SERVER_URL=http://127.0.0.1:19388 \
MOCR2_SERVED_MODEL_NAME=MonkeyOCRv2 \
  uv run --locked --extra vllm-cu126 uvicorn \
  --factory monkeyocr_gateway.app:create_app --host 127.0.0.1 --port 8080
```

Again, use `vllm-cu128` if that was your chosen extra. The served model name must match the vLLM server. Check the gateway and its connection to vLLM:

```sh
curl -fsS http://127.0.0.1:8080/health/ready
```

### 3. Parse an image or PDF

This example sends `docs/assets/vi-du-tieng-viet.pdf`, a synthetic Vietnamese document included in the repository. It base64-encodes the file locally and saves the response as `result.json`:

```sh
uv run --locked --extra vllm-cu126 python -c '
import base64, json, pathlib, sys
print(json.dumps({
    "file": base64.b64encode(pathlib.Path(sys.argv[1]).read_bytes()).decode(),
    "fileType": 0,
    "visualize": False,
    "returnMarkdownImages": False,
}))
' docs/assets/vi-du-tieng-viet.pdf |
curl -fsS http://127.0.0.1:8080/layout-parsing \
  -H 'Content-Type: application/json' --data-binary @- > result.json
```

Use `vllm-cu128` instead if selected. To parse another file, replace the sample path; set `"fileType": 0` for a PDF or `1` for an image. Each page's Markdown is in `result.layoutParsingResults[].markdown.text`. For example:

```sh
uv run --locked --extra vllm-cu126 python -c '
import json
with open("result.json") as f:
    result = json.load(f)
for page in result["result"]["layoutParsingResults"]:
    print(page["markdown"]["text"])
'
```

For WeKnora, configure the gateway's base URL (for example `http://127.0.0.1:8080` when running on the same host); the client appends `/layout-parsing`.

## Configuration and tests

Two-stage parsing is the default. Set `MOCR2_PARSING_MODE=end-to-end` to change the default, or send `vlmExtraArgs.parsingMode` per request. URL input is disabled unless its host and every redirect host are in `MOCR2_URL_ALLOWLIST`; base64 input does not require an allowlist. Document orientation and unwarping are not supported.

See [AGENTS.md](AGENTS.md) for environment settings and [docs/paddleocr-api.md](docs/paddleocr-api.md) for the HTTP contract.

```sh
uv run --locked --extra vllm-cu126 pytest
```

Use the same extra chosen above; alternatively, `uv sync` and `uv run pytest` install and test the gateway without the vLLM extra.
