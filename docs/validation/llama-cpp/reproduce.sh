#!/bin/sh
uv run --no-sync python scripts/compare_llama_cpp.py --vllm-gateway http://127.0.0.1:18081 --llama-gateway http://127.0.0.1:18082 --output docs/validation/llama-cpp docs/validation/paddle-format/input.png docs/validation/paddle-picture/input.png
