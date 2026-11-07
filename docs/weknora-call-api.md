## Điểm vào chính

Trong snapshot WeKnora `fd7a2586fd58`, code gọi PaddleOCR-VL self-hosted nằm ở **Go backend**, không phải Python `docreader/`.

Luồng chính:

```text
Chọn engine paddleocr_vl
  → docparser.NewReader(...)
  → NewPaddleOCRVLReader(...)
  → PaddleOCRVLReader.Read(...)
  → POST {endpoint}/layout-parsing
```

### 1. Gọi API và xử lý kết quả

Folder: [`internal/infrastructure/docparser/`](raw/WeKnora/internal/infrastructure/docparser/)

| File | Vai trò |
|---|---|
| [`paddleocr_vl_converter.go`](raw/WeKnora/internal/infrastructure/docparser/paddleocr_vl_converter.go) | **File quan trọng nhất.** `callLayoutParsing` tạo JSON, base64-encode file và gọi `POST /layout-parsing`; đọc response, ghép Markdown và ảnh theo trang. `Read` kiểm tra endpoint, xử lý bảng HTML và tạo kết quả. `PingPaddleOCRVL` dùng GET để kiểm tra khả năng kết nối. |
| [`paddleocr_vl_layout.go`](raw/WeKnora/internal/infrastructure/docparser/paddleocr_vl_layout.go) | Đọc layout block, bounding box và page từ response; căn chỉnh chúng với Markdown để tạo source locator. |
| [`html_table_normalizer.go`](raw/WeKnora/internal/infrastructure/docparser/html_table_normalizer.go) | Normalize bảng HTML trong Markdown trả về trước khi xử lý tiếp. |
| [`paddleocr_vl_converter_test.go`](raw/WeKnora/internal/infrastructure/docparser/paddleocr_vl_converter_test.go) | Test cấu hình timeout, response giả lập, deadline và cancellation. **Chưa chạy test trong phiên này.** |

Request gửi `file` dạng base64, `fileType` (`0` cho PDF, `1` cho ảnh), `visualize: false` và các recognition parameters. Endpoint cấu hình là **base URL**, vì code tự nối `/layout-parsing`. Code yêu cầu **full PaddleOCR-VL pipeline service**, không chỉ bare VLM inference server. Xem [`callLayoutParsing` và constructor](raw/WeKnora/internal/infrastructure/docparser/paddleocr_vl_converter.go).

### 2. Đăng ký và chọn parser

| File | Vai trò |
|---|---|
| [`engines.go`](raw/WeKnora/internal/infrastructure/docparser/engines.go) | Đăng ký engine `paddleocr_vl`, khai báo các định dạng PDF/JPG/JPEG/PNG/BMP/TIFF, kiểm tra availability và tạo reader. |
| [`engine_registry.go`](raw/WeKnora/internal/infrastructure/docparser/engine_registry.go) | `NewReader` tra engine đã đăng ký và gọi `NewReader` của engine đó. |
| [`knowledge_process.go`](raw/WeKnora/internal/application/service/knowledge_process.go) | Luồng knowledge ingestion: `resolveDocReader` chọn reader; `callDocReaderWithTimeout` gọi `reader.Read`. |
| [`temporary_document_parse.go`](raw/WeKnora/internal/application/service/temporary_document_parse.go) | Luồng tài liệu tạm: resolve parser rule, tạo reader và gọi `Read`. |
| [`session/attachment_processor.go`](raw/WeKnora/internal/handler/session/attachment_processor.go) | Luồng chat attachment cũng resolve reader qua registry và gọi `Read`; có fallback nếu registry không tạo được reader. |

### 3. Cấu hình endpoint và bảo mật

| File | Vai trò |
|---|---|
| [`internal/types/tenant.go`](raw/WeKnora/internal/types/tenant.go) | `ParserEngineConfig` chứa `paddleocr_vl_endpoint`, tùy chọn seal/chart recognition; `ToOverridesMap` chuyển cấu hình cho reader. |
| [`internal/application/service/parser_url_security.go`](raw/WeKnora/internal/application/service/parser_url_security.go) | Đưa endpoint PaddleOCR-VL vào danh sách URL override cần kiểm tra SSRF. Reader cũng kiểm tra SSRF trước khi gọi. |
| [`internal/handler/tenant.go`](raw/WeKnora/internal/handler/tenant.go) | Có validation SSRF cho endpoint trong cấu hình tenant. |
| [`frontend/src/views/settings/ParserEngineSettings.vue`](raw/WeKnora/frontend/src/views/settings/ParserEngineSettings.vue) | UI nhập và lưu cấu hình endpoint. |
| [`frontend/src/api/system/index.ts`](raw/WeKnora/frontend/src/api/system/index.ts) | Type frontend cho các trường cấu hình PaddleOCR-VL. |
| [`.env.example`](raw/WeKnora/.env.example) và [`docker-compose.yml`](raw/WeKnora/docker-compose.yml) | Cấu hình `WEKNORA_PADDLEOCR_VL_TIMEOUT`; mặc định trong converter là **1000 giây**. |

**Phân biệt:** [`paddleocr_vl_cloud_converter.go`](raw/WeKnora/internal/infrastructure/docparser/paddleocr_vl_cloud_converter.go) là nhánh **AI Studio cloud**, không phải client self-hosted `/layout-parsing`.

**Giới hạn kiểm tra:** Đây là danh sách các điểm liên quan đã tìm và đọc trực tiếp trong source, không khẳng định bao phủ mọi đường gọi gián tiếp. Chỉ kiểm tra tĩnh; chưa chạy service, request hay test. Wiki nền: [WeKnora document parsing](wiki/weknora-document-ingestion.md) và [source coverage](wiki/weknora-source-map.md). Kết quả được để transient, không sửa wiki.
