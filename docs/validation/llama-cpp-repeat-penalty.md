# llama-server repetition penalty — issue #17

The deployment that reported the error has not provided its llama.cpp version, build, launch command, or model. The following reproduces the same validation error on the local checkout, not on that deployment; it does not establish compatibility with other builds.

## Environment and result

- llama.cpp commit `ca8daad199a9c12c5b54643b974c640939a0f7bd`, build 12072, CUDA-enabled build, run with `-ngl 0` (CPU).
- `aenawi/MonkeyOCRv2-B-GGUF` Q8_0 and F16 mmproj, and synthetic `docs/assets/vi-du-tieng-viet.pdf`.
- A direct chat request with `repeat_penalty: 1.05` and `repeat_last_n: -1` returned HTTP 400 with the reported range error. Omitting `repeat_last_n` returned HTTP 200.
- After the gateway change, `/layout-parsing` with `repetitionPenalty: 1.05` returned HTTP 200 and `errorCode: 0` in both modes. Each response had one page and five blocks; Markdown contained 215 characters in two-stage and 217 in end-to-end. No gateway truncation warning was logged. This checks parsing, not OCR accuracy or penalty parity.
- The local run's server log, gateway log and four JSON responses are in `/tmp/monkeyocr-17-YCVnH5/` on the machine where it ran; that temporary directory is not a durable artifact.

The server schema in this commit restricts `repeat_last_n` to `0..INT32_MAX`; its default is 64 and 0 disables the penalty. The gateway now omits this field and uses the server's configured window. Consequently, the penalty need not cover the whole sequence as on vLLM. Large windows are not a free substitute: this commit's sampler constructs a ring buffer sized to the window. No claim is made about the unknown deployment or all other llama.cpp versions.

## Reproduce

From the repository root, with the model and gateway installed as in the README, start these in separate terminals (choose unused ports):

```sh
./llama.cpp/llama-server \
  -m models/MonkeyOCRv2-B-GGUF/MonkeyOCRv2-B.q8_0.gguf \
  --mmproj models/MonkeyOCRv2-B-GGUF/MonkeyOCRv2-B.mmproj-f16.gguf \
  -np 1 -c 8192 -ngl 0 -t 8 --alias MonkeyOCRv2 --host 127.0.0.1 --port 19417

MOCR2_SERVER_BACKEND=llama-cpp MOCR2_SERVER_URL=http://127.0.0.1:19417 \
  MOCR2_SERVED_MODEL_NAME=MonkeyOCRv2 MOCR2_REQUEST_TIMEOUT=300 \
  .venv/bin/uvicorn --factory monkeyocr_gateway.app:create_app --host 127.0.0.1 --port 18417
```

After the server is healthy, run the following; output files are written to the current directory:

```sh
uv run python - <<'PY'
import base64, json, pathlib, httpx

with httpx.Client(timeout=300) as client:
    url = 'http://127.0.0.1:19417/v1/chat/completions'
    base = {'model': 'MonkeyOCRv2', 'messages': [{'role': 'user', 'content': 'Hello'}],
            'max_tokens': 1, 'repeat_penalty': 1.05}
    for name, extra in [('negative', {'repeat_last_n': -1}), ('default', {})]:
        response = client.post(url, json=base | extra)
        pathlib.Path(f'{name}.json').write_text(response.text)
        print(name, response.status_code)
    data = base64.b64encode(pathlib.Path('docs/assets/vi-du-tieng-viet.pdf').read_bytes()).decode()
    for mode in ('two-stage', 'end-to-end'):
        body = {'file': data, 'fileType': 0, 'visualize': False, 'returnMarkdownImages': False,
                'repetitionPenalty': 1.05, 'maxNewTokens': 512, 'vlmExtraArgs': {'parsingMode': mode}}
        response = client.post('http://127.0.0.1:18417/layout-parsing', json=body)
        pathlib.Path(f'{mode}.json').write_text(response.text)
        print(mode, response.status_code, response.json().get('errorCode'))
PY
```
