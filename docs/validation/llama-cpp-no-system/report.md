# vLLM vs llama-cpp manual comparison

CER uses vLLM Markdown as reference; no pass/fail threshold is applied.

Inputs must be public or synthetic; do not use legal-vi-ocr output.

## 1. docs/validation/paddle-format/input.png

SHA-256: `03c6d6c089fd5625e52189561ec34021f24d54ab296636d028daa2c6e65b3a66`

### two-stage

```text
Markdown CER: 0.000000 (0 edits / 215 reference characters)
Blocks: vLLM=6, llama-cpp=6, delta=+0
Labels (vLLM -> llama-cpp, delta):
  doc_title: 1 -> 1 (+0)
  formula: 1 -> 1 (+0)
  paragraph_title: 2 -> 2 (+0)
  text: 2 -> 2 (+0)

Markdown diff:
(no differences)

Block label/content diff (reading order):
(no differences)
```

### end-to-end

```text
Markdown CER: 0.000000 (0 edits / 188 reference characters)
Blocks: vLLM=6, llama-cpp=6, delta=+0
Labels (vLLM -> llama-cpp, delta):
  doc_title: 1 -> 1 (+0)
  formula: 1 -> 1 (+0)
  paragraph_title: 2 -> 2 (+0)
  text: 2 -> 2 (+0)

Markdown diff:
(no differences)

Block label/content diff (reading order):
(no differences)
```

## 2. docs/validation/paddle-picture/input.png

SHA-256: `6c8297971db96b6d885e440f9b56b45605af59bf126c94be7f74be4ffdfa99e8`

### two-stage

```text
Markdown CER: 0.000000 (0 edits / 215 reference characters)
Blocks: vLLM=7, llama-cpp=7, delta=+0
Labels (vLLM -> llama-cpp, delta):
  doc_title: 1 -> 1 (+0)
  formula: 1 -> 1 (+0)
  image: 1 -> 1 (+0)
  paragraph_title: 2 -> 2 (+0)
  text: 2 -> 2 (+0)

Markdown diff:
(no differences)

Block label/content diff (reading order):
(no differences)
```

### end-to-end

```text
Markdown CER: 0.000000 (0 edits / 188 reference characters)
Blocks: vLLM=7, llama-cpp=7, delta=+0
Labels (vLLM -> llama-cpp, delta):
  doc_title: 1 -> 1 (+0)
  formula: 1 -> 1 (+0)
  image: 1 -> 1 (+0)
  paragraph_title: 2 -> 2 (+0)
  text: 2 -> 2 (+0)

Markdown diff:
(no differences)

Block label/content diff (reading order):
(no differences)
```
