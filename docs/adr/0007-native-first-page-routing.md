# Chế độ định tuyến theo trang (`parsingMode: "routed"`), ưu tiên Lớp chữ có sẵn

Nhiều PDF pháp lý có Lớp chữ có sẵn đúng từng ký tự. Chạy OCR trên các trang đó tốn Máy chủ suy luận và có thể làm sai chữ đã đúng. Dự án legal-vi-ocr đã xây và đánh giá router theo trang ở revision `6e7ad59` (`docs/page-router-architecture.md`, issue #12 bên đó). Gateway port router này thành giá trị thứ ba của `vlmExtraArgs.parsingMode`: `"routed"`. Giá trị này cũng dùng được cho `MOCR2_PARSING_MODE`. Mặc định vẫn là `two-stage`. Client như WeKnora không cần sửa gì, và request vẫn hợp lệ theo schema PaddleOCR-VL (ADR-0001).

Với mỗi trang PDF, quality gate (`router/gate.py`) quyết định `native` hoặc `ocr` từ nhiều tín hiệu: số ký tự, dấu tiếng Việt, glyph lỗi, chữ ẩn hoặc bị che, chữ ngoài trang, chữ trùng.

- `native`: Khối lấy từ PyMuPDF (chữ và bảng đã hậu xử lý). Mực ảnh mà chữ có sẵn không phủ được tách ra. Vùng giống chữ thành request OCR bổ sung: crop, che chữ có sẵn bằng màu trắng, thêm viền trắng nếu vùng nhỏ. Mực còn lại được giữ thành Khối `image` để người kiểm xem.
- `ocr`: cả trang đi qua Chế độ hai giai đoạn, như `parsingMode: "two-stage"`.
- Trang không có chữ và không đo được mực: không gọi OCR, giữ một Khối `image` rỗng để review.

Nhánh OCR luôn dùng Chế độ hai giai đoạn, theo ADR-0003 của legal-vi-ocr. Chỉ request bổ sung mới bật phục hồi chữ trong ảnh: Picture lớn có nét giống chữ được gửi lại với prompt Text của Upstream. Ảnh đầu vào (`fileType: 1`) không có Lớp chữ, nên đi thẳng nhánh OCR cả trang.

Response giữ contract PaddleOCR-VL và chỉ thêm trường:

- `prunedResult.routing`: quyết định, lý do, tín hiệu, ngưỡng, request OCR và `review_reasons`. Tọa độ trong đây là point PDF chưa xoay, tương đối CropBox.
- `parsing_res_list[].block_source` (`native` | `ocr` | `routing`) và `block_review`.
- `model_settings.page_router`.

Bảng có sẵn là HTML trong `block_content`. Khối có sẵn mang Nhãn Monkey `Text` hoặc `Table`.

## Considered Options

- Endpoint hoặc dịch vụ riêng: phải sửa client. Bị loại.
- Cờ riêng, độc lập với `parsingMode`, để nhánh OCR có thể là end-to-end: router đã được kiểm chứng với two-stage, còn phục hồi chữ trong ảnh và chống trùng cần layout của two-stage. Bị loại cho tới khi có đánh giá.
- Chạy batch runner của legal-vi-ocr. Runner này đọc manifest, kiểm ảnh chẩn đoán bằng hash, có prepare-only, có replay run đã ghi và ghi artifact ra đĩa. Gateway tự render trang, test dùng fake server, và không ghi đĩa. Vì vậy chỉ port các helper thuần, còn phần điều phối được viết lại bằng async (`routed.py`).

## Consequences

- Thêm dependency CPU: PyMuPDF và NumPy. Gateway vẫn không cần torch hay GPU (ADR-0003). PyMuPDF có giấy phép AGPL-3.0 hoặc thương mại. Cần xem lại giấy phép trước khi phân phối bản build đóng.
- Helper được port từ legal-vi-ocr `6e7ad59`. Mỗi file trong `router/` ghi rõ nguồn. `tests/test_router_parity.py` dùng lại các test tổng hợp của bên đó. Khi nâng revision, cần đối chiếu lại helper và các test này. Không copy fixture ghi từ corpus thật, vì đó là tài liệu nhạy cảm.
- Router khác legal-vi-ocr ở các điểm sau, có ghi trong docstring:
  - Một lần gọi suy luận lỗi làm cả request lỗi, như hai chế độ kia. Vì vậy không có trạng thái `pending` hay `error` theo trang.
  - Không theo dõi output bị cắt (`ocr_truncated`).
  - Trang OCR cả trang giữ thứ tự đọc của Monkey, thay vì sắp lại trên→dưới, trái→phải.
  - Mỗi trang được lập kế hoạch trên một bản sao tài liệu PyMuPDF riêng. Lý do: trên trang xoay, `find_tables()` ghi đè MediaBox và làm mất CropBox. Adapter bảng cũng gọi hàm này trên trang trước.
  - Trang có `/Rotate` khác 0 bỏ qua dò bảng và ghi `routing.native_tables_skipped = "rotated_page"`. Chữ trong bảng vẫn còn, dưới dạng Khối text. Lý do: `find_tables()` trả bảng theo trục đã xoay, còn adapter so với chữ theo trục chưa xoay. Thử trên trang tổng hợp, bảng bị loại nhưng không ghi `table_fallback`.
- Các ngưỡng là heuristic, không phải độ chính xác đã hiệu chỉnh. Cờ review không thay thế việc người kiểm đối chiếu bản chép với ảnh nguồn.
