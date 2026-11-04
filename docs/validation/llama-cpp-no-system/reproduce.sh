#!/bin/sh
uv run --no-sync python scripts/compare_llama_cpp.py --vllm-gateway http://127.0.0.1:18481 --llama-gateway http://127.0.0.1:18482 --output docs/validation/llama-cpp-no-system docs/validation/paddle-format/input.png docs/validation/paddle-picture/input.png
