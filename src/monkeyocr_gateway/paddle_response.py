"""Build PaddleOCR-VL Kết quả trang from Khối."""

from __future__ import annotations


def build_page_result(width: int, height: int, blocks: list[dict], model_settings: dict) -> dict:
    parsing_res_list = [
        {
            "block_label": block["label"],
            "block_content": block["content"],
            "block_bbox": block["bbox"],
            "block_id": index,
            "block_order": index + 1,
        }
        for index, block in enumerate(blocks)
    ]
    return {
        "prunedResult": {
            "width": width,
            "height": height,
            "model_settings": model_settings,
            "parsing_res_list": parsing_res_list,
        },
        "markdown": {
            "text": "\n\n".join(block["content"] for block in blocks if block["content"]),
            "images": {},
        },
        "outputImages": None,
        "inputImage": None,
    }
