"""Compare two running gateways on public or synthetic single-page images."""

import argparse
import base64
import difflib
import hashlib
import json
import shlex
import sys
from collections import Counter
from pathlib import Path

import httpx

MODES = ("two-stage", "end-to-end")


def cer(reference, candidate):
    """Character-level Levenshtein distance divided by reference length."""
    previous = list(range(len(candidate) + 1))
    for i, ref_char in enumerate(reference, 1):
        current = [i]
        for j, char in enumerate(candidate, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref_char != char))
            )
        previous = current
    distance = previous[-1]
    return distance, distance / len(reference) if reference else None


def diff(reference, candidate, name):
    lines = list(
        difflib.unified_diff(
            reference.splitlines(keepends=True),
            candidate.splitlines(keepends=True),
            fromfile=f"vllm/{name}",
            tofile=f"llama-cpp/{name}",
        )
    )
    if not lines:
        return "(no differences)\n"
    return "".join(line if line.endswith("\n") else line + "\n" for line in lines)


def page_data(response):
    if response.get("errorCode") != 0:
        raise ValueError(f"Gateway returned error: {response.get('errorMsg')}")
    pages = response["result"]["layoutParsingResults"]
    if len(pages) != 1:
        raise ValueError(f"Expected one page per image, got {len(pages)}")
    page = pages[0]
    return page["markdown"]["text"], [
        {"label": block["block_label"], "content": block["block_content"]}
        for block in page["prunedResult"]["parsing_res_list"]
    ]


def compare(reference, candidate):
    ref_md, ref_blocks = page_data(reference)
    candidate_md, candidate_blocks = page_data(candidate)
    distance, rate = cer(ref_md, candidate_md)
    ref_labels = Counter(block["label"] for block in ref_blocks)
    candidate_labels = Counter(block["label"] for block in candidate_blocks)
    lines = [
        f"Markdown CER: {rate:.6f} ({distance} edits / {len(ref_md)} reference characters)"
        if rate is not None
        else f"Markdown CER: undefined (empty reference; {distance} insertions)",
        f"Blocks: vLLM={len(ref_blocks)}, llama-cpp={len(candidate_blocks)}, "
        f"delta={len(candidate_blocks) - len(ref_blocks):+d}",
        "Labels (vLLM -> llama-cpp, delta):",
    ]
    for label in sorted(ref_labels.keys() | candidate_labels.keys()):
        lines.append(
            f"  {label}: {ref_labels[label]} -> {candidate_labels[label]} "
            f"({candidate_labels[label] - ref_labels[label]:+d})"
        )
    lines.extend(
        [
            "\nMarkdown diff:\n" + diff(ref_md, candidate_md, "markdown"),
            "Block label/content diff (reading order):\n"
            + diff(
                json.dumps(ref_blocks, ensure_ascii=False, indent=2) + "\n",
                json.dumps(candidate_blocks, ensure_ascii=False, indent=2) + "\n",
                "blocks",
            ),
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-gateway", required=True, help="Base URL of the vLLM gateway")
    parser.add_argument("--llama-gateway", required=True, help="Base URL of the llama-cpp gateway")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--timeout", type=float, default=1200, help="Seconds per request")
    parser.add_argument("pages", nargs="+", type=Path, help="Public/synthetic single-page images")
    args = parser.parse_args()

    inputs = [(path, path.read_bytes()) for path in args.pages]
    args.output.mkdir(parents=True, exist_ok=False)
    command = shlex.join(["uv", "run", "--no-sync", "python", *sys.argv])
    (args.output / "reproduce.sh").write_text(f"#!/bin/sh\n{command}\n", encoding="utf-8")
    report = [
        "# vLLM vs llama-cpp manual comparison\n",
        "CER uses vLLM Markdown as reference; no pass/fail threshold is applied.\n",
        "Inputs must be public or synthetic; do not use legal-vi-ocr output.\n",
    ]
    with httpx.Client(timeout=args.timeout, follow_redirects=False) as client:
        for index, (path, data) in enumerate(inputs, 1):
            encoded = base64.b64encode(data).decode("ascii")
            report.append(f"## {index}. {path}\n\nSHA-256: `{hashlib.sha256(data).hexdigest()}`\n")
            for mode in MODES:
                responses = []
                for backend, url in (
                    ("vllm", args.vllm_gateway),
                    ("llama-cpp", args.llama_gateway),
                ):
                    response = client.post(
                        url.rstrip("/") + "/layout-parsing",
                        json={
                            "file": encoded,
                            "fileType": 1,
                            "visualize": False,
                            "returnMarkdownImages": False,
                            "vlmExtraArgs": {"parsingMode": mode},
                        },
                    )
                    destination = args.output / f"{index:02d}-{mode}-{backend}.json"
                    destination.write_bytes(response.content)
                    response.raise_for_status()
                    responses.append(response.json())
                comparison = compare(*responses)
                report.append(f"### {mode}\n\n```text\n{comparison}```\n")
                print(f"{path} / {mode}\n{comparison}", flush=True)
                (args.output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Saved responses, report and command to {args.output}")


if __name__ == "__main__":
    main()
