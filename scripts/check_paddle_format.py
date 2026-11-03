"""Manual GPU check using synthetic input and a separate PaddleOCR environment."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--picture", action="store_true", help="Include a synthetic landscape")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    from PIL import Image, ImageDraw, ImageFont
    from paddleocr import PaddleOCRVL

    args.output.mkdir(parents=True, exist_ok=False)
    image = Image.new("RGB", (1600, 2000), "white")
    draw = ImageDraw.Draw(image)
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    title = ImageFont.truetype(str(font_dir / "DejaVuSans-Bold.ttf"), 60)
    heading = ImageFont.truetype(str(font_dir / "DejaVuSans-Bold.ttf"), 42)
    body = ImageFont.truetype(str(font_dir / "DejaVuSans.ttf"), 30)
    math = ImageFont.truetype(str(font_dir / "DejaVuSerif.ttf"), 64)
    draw.text((220, 160), "Synthetic Physics Notes", font=title, fill="black")
    draw.text((140, 360), "1. Energy", font=heading, fill="black")
    draw.text((140, 470), "The mass-energy relation is shown below.", font=body, fill="black")
    draw.text((560, 650), "E = mc²", font=math, fill="black")
    draw.text((1360, 675), "(1)", font=body, fill="black")
    draw.text((140, 950), "1.1 Explanation", font=heading, fill="black")
    draw.text((140, 1060), "Energy equals mass times the square of the speed of light.", font=body, fill="black")
    if args.picture:
        draw.rectangle((300, 1250, 1300, 1800), fill="#75bfff")
        draw.ellipse((1050, 1300, 1200, 1450), fill="#ffda42")
        draw.polygon([(300, 1700), (650, 1350), (1000, 1800), (300, 1800)], fill="#506b8a")
        draw.polygon([(600, 1800), (1000, 1450), (1300, 1750), (1300, 1800)], fill="#6a8860")
        draw.rectangle((300, 1750, 1300, 1800), fill="#38865a")
    image_path = args.output / "input.png"
    image.save(image_path)

    pipeline = PaddleOCRVL(
        pipeline_version="v1.6",
        layout_detection_model_dir=str(args.layout.resolve()),
        vl_rec_model_dir=str(args.model.resolve()),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_seal_recognition=False,
        engine="transformers",
        engine_config={"transformers": {"model_kwargs": {"dtype": "auto"}}},
        device="gpu:0",
    )
    results = list(pipeline.predict(str(image_path)))
    assert len(results) == 1, "Expected one page"
    result = results[0]
    raw = result.json
    markdown = result.markdown
    (args.output / "response.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "markdown.md").write_text(markdown["markdown_texts"], encoding="utf-8")
    if args.picture:
        assert markdown["markdown_images"], "No image in Paddle Markdown"
        for path, crop in markdown["markdown_images"].items():
            destination = args.output / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            crop.save(destination)
    versions = {name: importlib.metadata.version(name) for name in
                ("paddleocr", "paddlex", "transformers", "torch", "pillow")}
    (args.output / "versions.json").write_text(
        json.dumps(versions, indent=2) + "\n", encoding="utf-8"
    )
    labels = {block["block_label"] for block in raw["res"]["parsing_res_list"]}
    if args.picture:
        assert "image" in labels, f"No image detected: {labels}"
    assert labels & {"formula", "display_formula"}, f"No formula detected: {labels}"
    assert "doc_title" in labels, f"No document title detected: {labels}"
    assert "paragraph_title" in labels, f"No section heading detected: {labels}"
    print(f"Saved actual Paddle results to {args.output}")


if __name__ == "__main__":
    main()
