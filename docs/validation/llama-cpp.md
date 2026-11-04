# vLLM vs llama-cpp manual comparison — issue #16

This check compares two gateway HTTP responses, using vLLM Markdown as the CER reference. It is a drift measurement, not an accuracy score or a pass/fail threshold. It uses the two synthetic images generated for [paddle-format](paddle-format.md) and [paddle-picture](paddle-picture.md); no legal-vi-ocr output is used. Each input is a single-page image. Markdown images are disabled in both requests; block label and content comparisons include image blocks, but not crops or bounding boxes.

The [saved run](llama-cpp/) contains the exact comparison command (`reproduce.sh`), SHA-256 input hashes and results (`report.md`), and the eight raw gateway JSON responses. The script refuses to overwrite an output directory: when repeating the saved command, change `--output` to a new path. If a request fails, its response is saved before the script exits; earlier responses and any completed report sections remain in the output directory.

## Run and results

The run used NVIDIA A100 GPUs, vLLM 0.11.2 with PyTorch 2.9.0+cu126, and llama.cpp `ca8daad199a9c12c5b54643b974c640939a0f7bd` (CUDA build). The servers and gateways used their default generation and gateway settings except as shown below. The vLLM server was already running with the following command; both gateways and llama-server were started separately for this check. The GPU numbers are specific to this machine; select unused devices on another host.

```sh
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
  uv run --locked --extra vllm-cu126 python references/MonkeyOCRv2/parsing/serve.py \
  -m "$PWD/models/MonkeyOCRv2-B-Parsing" --host 127.0.0.1 -p 19388 \
  --served-model-name MonkeyOCRv2 --gpu-memory-utilization 0.15 \
  --max-model-len 16384 --max-num-batched-tokens 16384 --max-num-seqs 1

CUDA_VISIBLE_DEVICES=5 ./llama.cpp/llama-server \
  -m models/MonkeyOCRv2-B-GGUF/MonkeyOCRv2-B.q8_0.gguf \
  --mmproj models/MonkeyOCRv2-B-GGUF/MonkeyOCRv2-B.mmproj-f16.gguf \
  -np 4 -c 65536 -ngl 99 --alias MonkeyOCRv2 --host 127.0.0.1 --port 19389

MOCR2_SERVER_BACKEND=vllm MOCR2_SERVER_URL=http://127.0.0.1:19388 \
  MOCR2_SERVED_MODEL_NAME=MonkeyOCRv2 .venv/bin/uvicorn \
  --factory monkeyocr_gateway.app:create_app --host 127.0.0.1 --port 18081

MOCR2_SERVER_BACKEND=llama-cpp MOCR2_SERVER_URL=http://127.0.0.1:19389 \
  MOCR2_SERVED_MODEL_NAME=MonkeyOCRv2 .venv/bin/uvicorn \
  --factory monkeyocr_gateway.app:create_app --host 127.0.0.1 --port 18082
```

Run these in separate terminals after installing the models and environment as described in the [README](../../README.md), and wait for `/health/ready` on both gateways. Then run the command in `reproduce.sh` with a new output directory. The script requires `httpx` from the gateway environment and is not run by pytest.

| Synthetic input | Mode | Markdown CER | Blocks (vLLM / llama-cpp) | Label/content differences |
| --- | --- | ---: | ---: | --- |
| Text and formula | two-stage | 0 / 215 = 0 | 6 / 6 | None |
| Text and formula | end-to-end | 0 / 188 = 0 | 6 / 6 | None |
| Text, formula and picture | two-stage | 0 / 215 = 0 | 7 / 7 | None |
| Text, formula and picture | end-to-end | 0 / 188 = 0 | 7 / 7 | None |

**Limitation:** Both gateways logged `finish reason length` during each two-stage request, meaning at least one inference output in each request was truncated and retained as partial content. The gateway response does not identify the affected block. These identical outputs therefore do not establish complete recognition or general model equivalence. The end-to-end requests did not log this warning. CER is undefined for an empty reference; the report gives the insertion count instead. Label counts and ordered label/content diffs are reported independently of Markdown.
