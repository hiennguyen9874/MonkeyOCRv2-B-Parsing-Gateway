"""Pure helpers ported from Upstream MonkeyOCRv2.

Source: references/MonkeyOCRv2/parsing/core_runner.py at revision 106914e.
Upstream imports torch and its preprocessor at module level, so the helpers are
copied here instead of imported (ADR-0003). Keep them behaviourally identical;
tests/test_upstream_parity.py reuses Upstream's test cases to catch drift.
"""

from __future__ import annotations

import ast
import base64
import re
from io import BytesIO

from PIL import Image, ImageOps

UPSTREAM_REVISION = "106914e"

# core_runner.ALL_PROMPT, kept identical to Upstream. Upstream leaves Footnote
# and Picture without a prompt; the gateway adds Footnote back through
# GATEWAY_EXTRA_PROMPTS (ADR-0006).
ALL_PROMPT = {
    "Caption": "Please output the text content from the image.",
    # "Footnote": "Please output the text content from the image.",
    "List-item": "Please output the text content from the image.",
    "Page-footer": "Please output the text content from the image.",
    "Page-header": "Please output the text content from the image.",
    "Section-header": "Please output the text content from the image.",
    "Text": "Please output the text content from the image.",
    "Title": "Please output the text content from the image.",
    "Formula": "Please write out the expression of the formula in the image using LaTeX format.",
    "Table": "Please extract the table from the image and represent it in OTSL format.",
    "LAYOUT": "Please output the categories and coordinates of the document elements in reading order.",
    "END2END": "List the document elements in reading order, including their categories, coordinates, and the content of each element.",
}

# Gateway extension, not Upstream (ADR-0006, issue #19): Nhãn Monkey that
# Upstream's ALL_PROMPT leaves unrecognized but the gateway recognizes in
# two-stage mode. Upstream's training README lists Footnote among the labels that
# use the Text Recognition prompt, so the prompt is Upstream's own Text prompt.
GATEWAY_EXTRA_PROMPTS = {
    "Footnote": ALL_PROMPT["Text"],
}

# core_runner.get_layout and _recognize_one_block request limits.
LAYOUT_MIN_PIXELS = 1003520
LAYOUT_MAX_TOKENS = 4096
RECOGNITION_MAX_TOKENS = 5000


def image_to_png_data_uri(image: Image.Image) -> str:
    """core_runner.image_to_png_data_uri"""
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _extract_balanced_blocks(text: str, left: str, right: str):
    """core_runner._extract_balanced_blocks"""
    blocks = []
    depth = 0
    start = -1
    for index, char in enumerate(text):
        if char == left:
            if depth == 0:
                start = index
            depth += 1
        elif char == right and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                blocks.append(text[start:index + 1])
                start = -1
    return blocks


def _deduplicate_strings(values):
    """core_runner._deduplicate_strings"""
    return list(dict.fromkeys(values))


def _extract_tolerant_list_blocks(text: str):
    """core_runner._extract_tolerant_list_blocks"""
    blocks = _extract_balanced_blocks(text, "[", "]")
    first = text.find("[")
    if first != -1:
        tail = text[first:].strip()
        missing = tail.count("[") - tail.count("]")
        if tail and missing > 0:
            blocks.append(tail + ("]" * missing))
    return _deduplicate_strings(blocks)


def _extract_tolerant_dict_blocks(text: str):
    """core_runner._extract_tolerant_dict_blocks"""
    blocks = _extract_balanced_blocks(text, "{", "}")
    for index, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        end = None
        for cursor in range(index, len(text)):
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
                if depth == 0:
                    end = cursor + 1
                    break
        blocks.append(
            text[index:end]
            if end is not None
            else text[index:] + ("}" * max(depth, 1))
        )
    return _deduplicate_strings(blocks)


def _parse_tolerant_items(text: str, normalize_item):
    """core_runner._parse_tolerant_items"""
    text = (text or "").strip()
    if not text:
        return []

    def normalize_list(value):
        if not isinstance(value, list):
            return []
        return [item for raw in value if (item := normalize_item(raw)) is not None]

    try:
        complete = normalize_list(ast.literal_eval(text))
        if complete:
            return complete
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        pass

    best = []
    for block in _extract_tolerant_list_blocks(text):
        try:
            current = normalize_list(ast.literal_eval(block))
            if len(current) > len(best):
                best = current
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            continue

    dict_items = []
    for block in _extract_tolerant_dict_blocks(text):
        try:
            item = normalize_item(ast.literal_eval(block))
            if item is not None:
                dict_items.append(item)
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
            continue
    return dict_items if len(dict_items) > len(best) else best


def _map_bbox_to_image(bbox, width: int, height: int):
    """core_runner._map_bbox_to_image: 0-1000 model space -> integer pixels."""
    x1, y1, x2, y2 = bbox
    x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
    y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1 = max(0, min(int(round(x1)), width - 1 if width > 0 else 0))
    y1 = max(0, min(int(round(y1)), height - 1 if height > 0 else 0))
    x2 = max(x1 + 1, min(int(round(x2)), width))
    y2 = max(y1 + 1, min(int(round(y2)), height))
    return [x1, y1, x2, y2]


def _normalize_model_item(item, include_content: bool):
    """core_runner._normalize_model_item"""
    if not isinstance(item, dict) or "bbox" not in item or "label" not in item:
        return None
    bbox = item["bbox"]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        bbox = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    normalized = {
        "bbox": bbox,
        "label": item["label"] if isinstance(item["label"], str) else str(item["label"]),
    }
    if include_content:
        content = item.get("content", "")
        normalized["content"] = content if isinstance(content, str) else str(content or "")
    return normalized


def parse_layout_output(text: str, image_size: tuple[int, int]) -> list[dict]:
    """core_runner.get_layout, the per-page parsing done after batch_inference."""
    width, height = image_size
    items = _parse_tolerant_items(
        text,
        lambda item: _normalize_model_item(item, include_content=False),
    )
    return [{
        "bbox": _map_bbox_to_image(item["bbox"], width, height),
        "label": item["label"],
    } for item in items]


def _build_page_tasks(page_idx, image, layouts, doc_id=None):
    """core_runner._build_page_tasks: crops come from the full-resolution page."""
    width, height = image.size
    tasks = []
    for block_idx, item in enumerate(layouts):
        x1, y1, x2, y2 = item["bbox"]
        x1 = max(0, min(int(round(x1)), max(0, width - 1)))
        y1 = max(0, min(int(round(y1)), max(0, height - 1)))
        x2 = max(x1 + 1, min(int(round(x2)), width))
        y2 = max(y1 + 1, min(int(round(y2)), height))
        label = item["label"]
        task = {
            "image": image.crop((x1, y1, x2, y2)),
            "bbox": [x1, y1, x2, y2],
            "label": label,
            "question": ALL_PROMPT.get(label, ""),
            "need_infer": label in ALL_PROMPT,
            "page_idx": page_idx,
            "page_num": page_idx + 1,
            "block_idx": block_idx,
        }
        if doc_id is not None:
            task["doc_id"] = doc_id
        tasks.append(task)
    return tasks


def parse_end2end_output(text: str, image_size: tuple[int, int]) -> tuple[list[dict], list[dict]]:
    """core_runner.parse_end2end_output"""
    width, height = image_size
    records = []
    layouts = []
    items = _parse_tolerant_items(
        text,
        lambda item: _normalize_model_item(item, include_content=True),
    )
    for block_idx, item in enumerate(items):
        bbox = _map_bbox_to_image(item["bbox"], width, height)
        label = item["label"]
        records.append({
            "bbox": bbox,
            "label": label,
            "content": (item.get("content") or "").strip(),
            "_block_idx": block_idx,
        })
        layouts.append({"bbox": bbox, "label": label})
    return records, layouts


def otsl_to_html(otsl_str):
    """core_runner.otsl_to_html"""
    if not otsl_str or not otsl_str.strip():
        return "<table></table>"
    if re.search(r'<otsl>', otsl_str, flags=re.I):
        return re.sub(
            r'<otsl>(.*?)</otsl>',
            lambda match: otsl_to_html(match.group(1)),
            otsl_str,
            flags=re.I | re.S,
        )
    
    rows_tokens = otsl_str.split("<nl>")
    if rows_tokens and rows_tokens[-1] == "":
        rows_tokens.pop()
    
    grid = []
    
    for r_idx, row_str in enumerate(rows_tokens):
        if not row_str.strip():
            if r_idx >= len(grid):
                grid.append([])
            continue
        
        # Only OTSL control tags are tokens. Other markup (e.g. <br> or a
        # nested <table>) is cell content and must remain untouched.
        parts = re.findall(
            r'<(fcel|ecel|lcel|ucel|xcel|nl)>(.*?)(?=<(?:fcel|ecel|lcel|ucel|xcel|nl)>|\Z)',
            row_str, flags=re.DOTALL | re.IGNORECASE,
        )
        
        if r_idx >= len(grid):
            grid.append([])
        
        col_idx = 0
        
        for tag, content in parts:
            while True:
                while len(grid[r_idx]) <= col_idx:
                    grid[r_idx].append(None)
                
                if grid[r_idx][col_idx] is not None:
                    col_idx += 1
                else:
                    break
            
            if tag == 'fcel' or tag == 'ecel':
                # Do not normalize whitespace. Decode the private OTSL escape
                # used by html2otsl for literal control-looking tags.
                text = content.replace("\ue100\ue100", "\ue100")
                text = re.sub(r"\ue100(fcel|ecel|lcel|ucel|xcel|nl)>", r"<\1>", text, flags=re.I)
                if tag == 'ecel':
                    text = ""
                grid[r_idx][col_idx] = {
                    'text': text,
                    'rowspan': 1,
                    'colspan': 1,
                    'valid': True
                }
                col_idx += 1
                
            elif tag == 'lcel':
                search_c = col_idx - 1
                found = False
                while search_c >= 0:
                    if len(grid[r_idx]) > search_c:
                        cell = grid[r_idx][search_c]
                        if cell and cell.get('valid'):
                            cell['colspan'] += 1
                            found = True
                            break
                    search_c -= 1
                
                if found:
                    grid[r_idx][col_idx] = {'valid': False, 'type': 'lcel'}
                else:
                    grid[r_idx][col_idx] = {
                        'text': '',
                        'rowspan': 1,
                        'colspan': 1,
                        'valid': True
                    }
                col_idx += 1
                
            elif tag == 'ucel':
                search_r = r_idx - 1
                found = False
                while search_r >= 0:
                    if len(grid[search_r]) > col_idx:
                        cell = grid[search_r][col_idx]
                        if cell and cell.get('valid'):
                            cell['rowspan'] += 1
                            found = True
                            break
                    search_r -= 1
                
                if found:
                    grid[r_idx][col_idx] = {'valid': False, 'type': 'ucel'}
                else:
                    grid[r_idx][col_idx] = {
                        'text': '',
                        'rowspan': 1,
                        'colspan': 1,
                        'valid': True
                    }
                col_idx += 1
                
            elif tag == 'xcel':
                grid[r_idx][col_idx] = {'valid': False, 'type': 'xcel'}
                col_idx += 1
            else:
                col_idx += 1
    
    html_parts = ['<table>']
    
    for row in grid:
        html_parts.append('<tr>')
        for cell in row:
            if cell is None:
                continue
            elif cell.get('valid'):
                attrs = []
                if cell['rowspan'] > 1:
                    attrs.append(f'rowspan="{cell["rowspan"]}"')
                if cell['colspan'] > 1:
                    attrs.append(f'colspan="{cell["colspan"]}"')
                
                attr_str = ' ' + ' '.join(attrs) if attrs else ''
                # Cell content is already serialized HTML from html2otsl;
                # preserve tags, entities and leading/trailing whitespace.
                raw_text = cell['text'].replace('\r\n', '\n').replace('\r', '\n')
                # Preserve embedded HTML tags/entities while escaping plain
                # text, and represent OTSL text newlines as HTML line breaks.
                chunks = re.split(r'(<[A-Za-z][^>]*>)', raw_text)
                rendered = []
                for i, chunk in enumerate(chunks):
                    if i % 2:
                        rendered.append('<br>' if re.fullmatch(r'<br\s*/?>', chunk, flags=re.I) else chunk)
                    else:
                        chunk = re.sub(r'&(?!(?:#\d+|#x[0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]+);)', '&amp;', chunk)
                        chunk = chunk.replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
                        rendered.append(chunk.replace('\n', '<br>'))
                text = ''.join(rendered)
                html_parts.append(f'<td{attr_str}>{text}</td>')
        html_parts.append('</tr>')
    
    html_parts.append('</table>')
    return ''.join(html_parts)


def process_formula(content: str):
    """core_runner.process_formula: returns (LaTeX, trailing equation number or None)."""
    content = content.strip('$').strip()
    # Collapse repeated \quad sequences (>=5).
    content = re.sub(
        r'(?:\\quad\s*){5,}',
        r'\\quad ',
        content
    )

    # Collapse repeated \qquad sequences (>=5).
    content = re.sub(
        r'(?:\\qquad\s*){5,}',
        r'\\qquad ',
        content
    ).strip()

    # Extract trailing (xxx). TODO: remove tag{}.
    match = re.search(
        r'(?:\\quad|\\qquad|\\eqno)\s*\(([^()]*)\)\s*$'
        r'|\\tag\{([^{}]*)\}\s*$',
        content
    )

    extracted = None
    if match:
        extracted = match.group(1)
        content = content[:match.start()].rstrip()

    begin_env = None
    # Detect leading \begin{xx}.
    begin_match = re.match(r'^\\begin\{([^\}]+)\}', content)
    if begin_match:
        begin_env = begin_match.group(1)
        # Remove leading \begin{xx}.
        content = content[begin_match.end():].lstrip()

        # Detect whether the matching \end{xx} is at the end.
        end_pattern = rf'\\end\{{{re.escape(begin_env)}\}}\s*$'
        end_match = re.search(end_pattern, content)
        if end_match:
            # Remove trailing \end{xx}.
            content = content[:end_match.start()].rstrip()


    # Extract trailing (xxx).
    match = re.search(
        r'(?:\\quad|\\qquad|\\eqno)\s*\(([^()]*)\)\s*$'
        r'|\\tag\{([^{}]*)\}\s*$',
        content
    )

    if match:
        extracted = match.group(1)
        content = content[:match.start()].rstrip()

    # ===== Restore begin/end =====

    if begin_env:
        content = f"\\begin{{{begin_env}}}\n{content}\n\\end{{{begin_env}}}"

    return content, extracted


def detect_repeat_token(
    predicted_tokens: str,
    base_max_repeats: int = 4,
    window_size: int = 500,
    cut_from_end: int = 0,
    scaling_factor: float = 3.0,
):
    if cut_from_end > 0:
        predicted_tokens = predicted_tokens[:-cut_from_end]

    if not predicted_tokens:
        return False

    max_seq_len = min(window_size // 2, len(predicted_tokens))
    for seq_len in range(1, max_seq_len + 1):
        candidate_seq = predicted_tokens[-seq_len:]
        max_repeats = int(base_max_repeats * (1 + scaling_factor / seq_len))
        if predicted_tokens.endswith(candidate_seq * (max_repeats + 1)):
            return True

    return False


def _should_retry_repeat_output(raw: str) -> bool:
    raw = raw or ""
    return detect_repeat_token(raw) or (
        len(raw) > 50 and detect_repeat_token(raw, cut_from_end=50)
    )


def repeat_retry_sampling(retry: int) -> tuple[float, float]:
    """(temperature, top_p) of the retry-th repeat retry (0-based), as in
    _recognize_one_block and batch_inference_with_repeat_retry."""
    return min(0.2 * (retry + 1), 0.8), 0.95


def _apply_jpeg_orientation(img: Image.Image) -> Image.Image:
    """core_runner._apply_jpeg_orientation, without the path-based JPEG check
    (the gateway never loads from a path or URL string)."""
    if (getattr(img, "format", None) or "").upper() == "JPEG":
        return ImageOps.exif_transpose(img)
    return img


def load_image(image: Image.Image, max_pixels: int | None = None, min_pixels: int | None = None) -> Image.Image:
    """core_runner.load_image, restricted to the PIL-image branch and the
    max_pixels/min_pixels arguments that MonkeyOCRv2_ServerParsing.batch_inference uses.
    """
    try:
        img = _apply_jpeg_orientation(image)
        img = img.convert("RGB")

        # resize image if too small
        if min_pixels and img.size[0] * img.size[1] < min_pixels:
            scale = (min_pixels / (img.size[0] * img.size[1])) ** 0.5
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.LANCZOS)

        # resize image if too large
        if max_pixels and img.size[0] * img.size[1] > max_pixels:
            scale = (max_pixels / (img.size[0] * img.size[1])) ** 0.5
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.LANCZOS)

        if max(img.size[0], img.size[1]) / min(img.size[0], img.size[1]) > 200:
            img = Image.new("RGB", (32, 32))
    except Exception:
        # Upstream silently substitutes a dummy image for anything it cannot load.
        img = Image.new("RGB", (32, 32))
    return img
