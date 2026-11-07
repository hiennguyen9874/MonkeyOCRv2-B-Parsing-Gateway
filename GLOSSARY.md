# MonkeyOCRv2-B-Parsing Gateway

Dịch vụ đặt trước MonkeyOCRv2-B-Parsing để client của PaddleOCR-VL gọi được với cùng API.

## Chế độ phân tích

**Chế độ hai giai đoạn** (two-stage mode):
Cách phân tích trang gồm một lần suy luận bố cục cho cả trang, sau đó nhận dạng riêng từng Khối đã cắt từ ảnh trang.
_Avoid_: layout mode, pipeline mặc định

**Chế độ end-to-end** (end-to-end mode):
Cách phân tích trang bằng một lần suy luận duy nhất trả về đồng thời loại, tọa độ và nội dung của mọi Khối.
_Avoid_: end2end, single-pass

**Chế độ định tuyến theo trang** (routed mode):
Cách phân tích PDF quyết định riêng từng trang: dùng Lớp chữ có sẵn khi qua quality gate, chỉ gửi trang hoặc vùng ảnh không tin được sang Chế độ hai giai đoạn.
_Avoid_: hybrid mode, native mode

**Lớp chữ có sẵn** (embedded text layer):
Chữ nhúng trong PDF mà PyMuPDF trích được, không qua OCR; có thể ẩn, hỏng hoặc bị che nên phải qua quality gate.
_Avoid_: native text (khi nói chung), text PDF

**OCR bổ sung** (supplemental OCR):
Request OCR cho một vùng ảnh trên trang đã qua quality gate, với chữ có sẵn đã bị che trắng; kết quả luôn cần review.
_Avoid_: partial OCR

## Trang và Khối

**Khối**:
Một vùng trên trang có nhãn loại, tọa độ và nội dung.
_Avoid_: block, region, element

**Kết quả trang**:
Kết quả phân tích của đúng một trang đầu vào, gồm danh sách Khối theo thứ tự đọc và Markdown của trang.
_Avoid_: page result, layout result

**Nhãn Monkey**:
Tên loại Khối do MonkeyOCRv2 sinh ra (ví dụ `Text`, `Section-header`, `Page-header`).
_Avoid_: label gốc, category

**Nhãn Paddle**:
Tên loại Khối theo từ vựng PaddleOCR-VL mà client nhận (ví dụ `text`, `paragraph_title`, `header`).
_Avoid_: block type, label chuẩn

## Suy luận

**Máy chủ suy luận** (inference server):
Máy chủ tương thích OpenAI chạy mô hình MonkeyOCRv2 mà gateway gửi yêu cầu suy luận tới; là vLLM hoặc llama-server.
_Avoid_: vLLM (khi muốn nói chung cho cả hai), backend, engine

## Nguồn tham chiếu

**Upstream**:
Bản cài đặt MonkeyOCRv2 tham chiếu tại revision đã pin; là chuẩn hành vi suy luận mà gateway phải khớp.
_Avoid_: fork, reference code (khi muốn nói chuẩn hành vi)
