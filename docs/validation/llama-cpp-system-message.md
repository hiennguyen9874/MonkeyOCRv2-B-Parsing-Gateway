# llama-server without a system message — issue #18

The gateway used to prepend `You are a helpful assistant.` as a system message to every llama-cpp inference call. It now sends only the user message on both backends. This note records why, and the manual check on a real llama-server.

## Origin of the system line

| Source (Upstream `references/MonkeyOCRv2` at `106914e`, model `zenosai/MonkeyOCRv2-B-Parsing`) | Default system |
| --- | --- |
| ms-swift training template `monkeyocrv2`, registered with `default_system=None` (`parsing/train/ms-swift/swift/llm/template/template/monkeyocrv2.py`); the model's `args.json` has `"template": "monkeyocrv2"` and `"system": null` | None |
| Model `chat_template.jinja`: emits a system block only when `messages[0].role == 'system'` | None |
| `tokenizer.chat_template` in `MonkeyOCRv2-B.q8_0.gguf`: byte-identical to `chat_template.jinja` | None |
| Upstream server path `parsing/core_runner.py::_chat_completion` (used by `parse.py`, FastAPI, Gradio) and the HF path in `parsing/cpu/core_runner.py` (`apply_chat_template`) | None |
| `parsing/cpu/core_runner.py::build_vllm_prompt` (offline vLLM, CPU variant only) | Hard-coded |

The GGUF model card recommends the line because it follows `build_vllm_prompt`; the conversion does not need it. The training datasets are private, so per-sample system fields cannot be ruled out, but Upstream's main inference path never sends one.

## Prompt rendering on llama-server

llama.cpp `ca8daad199a9c12c5b54643b974c640939a0f7bd` (build 12072) uses the GGUF's Jinja template by default. `POST /apply-template` with an image part and text `Q`:

```text
no system:   <|im_start|>user\n<__media_…__>Q<|im_end|>\n<|im_start|>assistant\n
with system: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<__media_…__>Q<|im_end|>\n<|im_start|>assistant\n
```

For the `dots_ocr` projector, mtmd wraps the image embeddings in `<|img|>` … `<|endofimg|>`. `POST /tokenize` maps these to ids 151652 and 151653, which are the original `<|vision_start|>` and `<|vision_end|>` ids. The GGUF renames these two tokens. Without the system message, the token sequence therefore has the same layout as the vLLM chat template and the training template. Do not start llama-server with `--no-jinja` or a `--chat-template` override.

## Comparison run

Same hardware, versions, inputs and server flags as [#16](llama-cpp.md): A100, vLLM 0.11.2 with PyTorch 2.9.0+cu126, llama.cpp `ca8daad19` CUDA build, and the two synthetic images. The [saved run](llama-cpp-no-system/) has the command, the report and the raw responses. The #16 run in [llama-cpp/](llama-cpp/) was made with the system message, so it serves as the "before" result.

| Input / mode | vLLM same as #16 | llama-cpp with system = vLLM | llama-cpp without system = vLLM |
| --- | --- | --- | --- |
| Text and formula / two-stage | yes | yes | yes |
| Text and formula / end-to-end | yes | yes | yes |
| Text, formula and picture / two-stage | yes | **no**: one bbox `y2` 506 vs 504 | yes |
| Text, formula and picture / end-to-end | yes | yes | yes |

"=" compares label, content and bbox of every block in order. Markdown CER was 0 in all four cases, both before and after. With the system message, the only difference was one bbox coordinate. Without it, all eight responses matched vLLM exactly.

**Limitation:** this is two synthetic pages. As in #16, every two-stage request logged `finish reason length` on both backends, so some content is truncated. The result shows that removing the line moves llama-cpp toward vLLM on this sample. It is not an accuracy benchmark or a proof of general equivalence.
