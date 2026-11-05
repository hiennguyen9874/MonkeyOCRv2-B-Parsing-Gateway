# Nhận dạng Footnote trong Chế độ hai giai đoạn

Ở revision đã pin `106914e`, `ALL_PROMPT` của Upstream comment dòng `Footnote`, nên Chế độ hai giai đoạn của Upstream giữ bbox của Khối `Footnote` nhưng để nội dung rỗng. Gateway ban đầu port y nguyên hành vi này (spec #1, user story 43; ticket #3). Issue #19 cho thấy hành vi đó chặn WeKnora: client bỏ `footnote` khỏi `markdownIgnoreLabels` nhưng vẫn không nhận được text chú thích.

Gateway cố ý lệch khỏi Upstream ở đúng một điểm: Khối `Footnote` được nhận dạng trong Chế độ hai giai đoạn bằng prompt Text của Upstream (`ALL_PROMPT["Text"]`), với cùng crop, giới hạn 5.000 token và repeat retry như mọi Khối Text. Lý do:

- WeKnora và contract PaddleOCR-VL là mục tiêu nghiệm thu. PaddleOCR-VL nhận dạng mọi Khối chữ, kể cả `footnote`, rồi `markdownIgnoreLabels` chỉ lọc lúc ghép Markdown; `block_content` vẫn có nội dung (output PaddleOCR-VL thật trong legal-vi-ocr: 5/5 Khối `footnote` có text).
- Model được train cho nhãn này: `references/MonkeyOCRv2/parsing/train/README.md` liệt kê `Footnote` trong các nhãn dùng prompt Text Recognition.
- Chế độ end-to-end vốn đã trả nội dung Footnote; thay đổi này làm hai chế độ nhất quán.

Cách làm giữ phần port nguyên vẹn: `ALL_PROMPT` và `_build_page_tasks` vẫn giống Upstream và vẫn có test parity. Ngoại lệ nằm trong `GATEWAY_EXTRA_PROMPTS` (`upstream.py`), được page parser áp lên các task sau `_build_page_tasks`. Test parity khóa đúng nội dung của ngoại lệ này. `Picture` vẫn không nhận dạng; nó được crop vào `markdown.images` (#7).

Footnote luôn được nhận dạng, kể cả khi client bỏ `footnote` khỏi Markdown (đây cũng là mặc định khi `markdownIgnoreLabels: null`), để `parsing_res_list` giữ contract giống PaddleOCR-VL. Không có biến môi trường để tắt.

## Consequences

Mỗi Khối Footnote tốn thêm một lần gọi Máy chủ suy luận. Khi nâng revision Upstream, nếu Upstream bật lại `Footnote` trong `ALL_PROMPT`, test parity của `GATEWAY_EXTRA_PROMPTS` sẽ fail và ngoại lệ này cần được gỡ bỏ.
