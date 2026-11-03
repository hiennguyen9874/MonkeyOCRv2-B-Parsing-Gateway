# Mô tả API tham chiếu PaddleOCR 3.x

## 1. Phạm vi và nguồn

Tài liệu mô tả **HTTP API phục vụ suy luận** trong `references/PADDLEOCR`, đặc biệt là PaddleOCR-VL HPS gateway. Đây là mô tả tham chiếu, **không khẳng định gateway MonkeyOCR trong repo này đã triển khai các endpoint này**. Các pipeline PaddleOCR/PaddleX có thể chạy ở các service khác nhau: cùng đường dẫn không có nghĩa cùng model hoặc cùng tập tham số.

- [Code HPS gateway: route, health, chuyển tiếp và xử lý lỗi](../references/PADDLEOCR/deploy/paddleocr_vl_docker/hps/gateway/app.py#L27-L413).
- [Kiến trúc và triển khai HPS](../references/PADDLEOCR/deploy/paddleocr_vl_docker/hps/README_en.md#L1-L143): client → FastAPI → Triton → VLM. HPS chỉ expose các route nêu trong mục 2; SDK/model repository của Triton được tải từ PaddleX khi chuẩn bị triển khai, không nằm trong source gateway này.
- [Code wrapper PaddleOCR-VL](../references/PADDLEOCR/paddleocr/_pipelines/paddleocr_vl.py#L30-L224): tham số predict và restructure; implementation pipeline thực tế được ủy quyền cho PaddleX.
- [Đặc tả serving PaddleOCR-VL](../references/PADDLEOCR/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md#L2210-L2715) là nguồn cho schema chi tiết mà gateway chỉ nhận dưới dạng `dict`.
- API SDK gọi **dịch vụ hosted** là API riêng, không mặc định tương thích với self-hosted HTTP API này: [SDK README](../references/PADDLEOCR/api_sdk/README.md), [Python client](../references/PADDLEOCR/paddleocr/_api_client/client.py).

Các liên kết dòng trong tài liệu trỏ đến bản reference tại thời điểm khảo sát; phiên bản PaddleX, pipeline và cấu hình triển khai quyết định hành vi thực tế.

## 2. Danh mục endpoint

| Method | Path | Service/pipeline | Mục đích |
|---|---|---|---|
| POST | `/layout-parsing` | PaddleOCR-VL HPS; cũng có ở PP-StructureV3 serving | Phân tích bố cục, OCR và xuất Markdown |
| POST | `/restructure-pages` | PaddleOCR-VL HPS | Hậu xử lý kết quả nhiều trang |
| GET | `/health` | PaddleOCR-VL HPS | Liveness của gateway |
| GET | `/health/ready` | PaddleOCR-VL HPS | Readiness của Triton, hai model và VLM |
| POST | `/ocr` | OCR serving riêng | Phát hiện và nhận dạng chữ |
| POST | `/chatocr-visual`, `/chatocr-vector`, `/chatocr-mllm`, `/chatocr-chat` | PP-ChatOCRv4 serving riêng | Trích xuất thông tin, vector và hỏi đáp |
| POST | `/doctrans-visual`, `/doctrans-translate` | PP-DocTranslation serving riêng | Phân tích và dịch tài liệu |

Những endpoint ở ba dòng cuối **không được đăng ký trong HPS gateway**. Đây không phải danh sách toàn bộ API model/pipeline của PaddleX. FastAPI cũng có thể cung cấp OpenAPI/docs mặc định; đó không phải route nghiệp vụ của pipeline.

## 3. Quy ước chung

Các POST bên dưới dùng JSON (`Content-Type: application/json`), tên trường theo camelCase. Đầu vào `file` là URL mà **server** truy cập được hoặc nội dung file mã hóa Base64, không phải đường dẫn local trên máy client. `fileType: 0` là PDF, `1` là ảnh, gồm TIFF nhiều trang; nếu không cung cấp, tài liệu nói suy ra từ URL. Khi dùng Base64 nên truyền `fileType` tường minh. Số trang thực xử lý có thể bị giới hạn bởi cấu hình service.

Với API serving được tài liệu hóa, response thành công có dạng:

```json
{"logId":"<id>", "errorCode":0, "errorMsg":"Success", "result":{}}
```

`errorMsg` cụ thể phụ thuộc service. Khi lỗi, response thường gồm `logId`, `errorCode`, `errorMsg`, không có `result`. Không giả định mọi HTTP service hoặc lỗi proxy đều theo schema này.

Riêng HPS, gateway chuyển body JSON đến Triton và thêm `logId` nếu chưa có; caller có thể cung cấp `logId` để tương quan request. Gateway không tự kiểm tra schema nghiệp vụ ngoài yêu cầu body là object; validation thực tế thuộc backend. Lỗi timeout trả HTTP 504, lỗi Triton/ngoại lệ trả 500, backend trả `errorCode != 0` thì gateway dùng mã đó làm HTTP status. Handler còn định nghĩa 400 cho JSONDecodeError và 422 cho RequestValidationError; JSON sai cú pháp có thể được FastAPI phân loại thành lỗi validation 422. Tham khảo [code xử lý](../references/PADDLEOCR/deploy/paddleocr_vl_docker/hps/gateway/app.py#L216-L407).

## 4. POST /layout-parsing — PaddleOCR-VL

[Route HPS](../references/PADDLEOCR/deploy/paddleocr_vl_docker/hps/gateway/app.py#L327-L341) chuyển request tới model Triton `layout-parsing`. [Schema đầy đủ](../references/PADDLEOCR/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md#L2285-L2590). Tham số bên dưới là schema được tài liệu PaddleOCR-VL mô tả; không suy ra tất cả phiên bản SDK tải về đều hỗ trợ như nhau.

### Request

Chỉ `file` là bắt buộc; các trường khác là tùy chọn. Các trường nullable có thể bỏ qua hoặc gửi null để dùng hành vi cấu hình pipeline.

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `file` | string | URL hoặc Base64 của PDF/ảnh |
| `fileType` | integer/null | 0 PDF; 1 ảnh |
| `useDocOrientationClassify`, `useDocUnwarping` | boolean/null | Chỉnh hướng và độ cong tài liệu |
| `useLayoutDetection` | boolean/null | Bật phát hiện bố cục |
| `useChartRecognition`, `useSealRecognition`, `useOcrForImageBlock` | boolean/null | Nhận dạng biểu đồ, con dấu, OCR trong khối ảnh |
| `layoutThreshold` | number/object/null | Ngưỡng phát hiện layout |
| `layoutNms` | boolean/null | NMS cho layout |
| `layoutUnclipRatio` | number/array/object/null | Tỷ lệ mở rộng box |
| `layoutMergeBboxesMode` | string/object/null | Cách gộp box |
| `layoutShapeMode` | string | Chế độ hình dạng layout |
| `promptLabel` | string/null | Nhãn prompt VLM |
| `formatBlockContent`, `mergeLayoutBlocks` | boolean/null | Định dạng nội dung/gộp khối layout |
| `repetitionPenalty`, `temperature`, `topP` | number/null | Tham số sinh VLM |
| `minPixels`, `maxPixels`, `maxNewTokens` | number/null | Giới hạn ảnh đầu vào và token sinh |
| `markdownIgnoreLabels` | array/null | Nhãn bị bỏ qua khi tạo Markdown |
| `vlmExtraArgs` | object/null | Tham số bổ sung cho VLM |
| `prettifyMarkdown`, `showFormulaNumber` | boolean | Làm đẹp Markdown mặc định true; hiển thị số công thức mặc định false |
| `restructurePages` | boolean | Tái cấu trúc các trang trong lúc infer; mặc định false |
| `mergeTables`, `relevelTitles` | boolean | Gộp bảng/đánh lại cấp tiêu đề; chỉ có hiệu lực khi `restructurePages=true` |
| `returnMarkdownImages` | boolean | Mặc định true; false bỏ qua encode/upload ảnh Markdown, `images` null hoặc vắng mặt |
| `outputFormats` | array/null | Định dạng xuất thêm; hiện tài liệu chỉ nêu `["docx"]` |
| `visualize` | boolean/null | Trả ảnh trung gian/kết quả; null hoặc bỏ qua dùng `Serving.visualize`, nếu không cấu hình thì mặc định trả ảnh |

Các ngưỡng, giá trị hợp lệ và default của model phụ thuộc pipeline/config; xem [predict wrapper](../references/PADDLEOCR/paddleocr/_pipelines/paddleocr_vl.py#L103-L216) và [tài liệu tham số](../references/PADDLEOCR/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md).

Ví dụ (thay URL bằng tài nguyên server truy cập được):

```bash
curl -sS http://localhost:8080/layout-parsing \
  -H 'Content-Type: application/json' \
  -d '{"file":"https://example.com/document.pdf","fileType":0,"visualize":false,"returnMarkdownImages":false}'
```

### Response

`result.dataInfo` là metadata đầu vào; `result.layoutParsingResults` là mảng kết quả từng trang thực xử lý (ảnh thông thường một phần tử; PDF/TIFF nhiều trang có thể có nhiều phần tử). Mỗi phần tử gồm:

| Trường | Ý nghĩa |
|---|---|
| `prunedResult` | Object kết quả predict rút gọn, bỏ `input_path` và `page_index`; cấu trúc chi tiết phụ thuộc pipeline/model |
| `markdown.text` | Chuỗi Markdown |
| `markdown.images` | Object ánh xạ đường dẫn tương đối tới dữ liệu ảnh; có thể null/vắng mặt |
| `outputImages` | Object ảnh trực quan/trung gian, có thể null |
| `inputImage` | Ảnh đầu vào JPEG, có thể null |
| `exports` | Object xuất thêm, ví dụ `{"docx":{"content":"..."}}`, khi yêu cầu |

Các trường dữ liệu nhị phân mặc định là Base64, nhưng có thể là URL ký sẵn nếu server cấu hình trả URL; không nên luôn giải mã Base64. Response minh họa tối giản, không phải fixture cố định:

```json
{
  "logId": "<id>", "errorCode": 0, "errorMsg": "Success",
  "result": {
    "dataInfo": {},
    "layoutParsingResults": [
      {"prunedResult": {}, "markdown": {"text": "# Nội dung", "images": null}}
    ]
  }
}
```

### Biến thể PP-StructureV3

PP-StructureV3 cũng dùng `POST /layout-parsing`, nhưng là pipeline khác. Ngoài các trường chung, có `useTextlineOrientation`, `useTableRecognition`, `useFormulaRecognition`, `useRegionDetection`, các tham số `textDet*`, `textRecScoreThresh`, `sealDet*`, `sealRecScoreThresh` và các tùy chọn nhận dạng bảng. Không gửi mặc định các trường VLM-only như `temperature` hay `promptLabel` sang PP-StructureV3. Mặc định tài liệu nêu giới hạn 10 trang PDF/TIFF, có thể đổi `Serving.extra.max_num_input_imgs`. Xem [schema PP-StructureV3](../references/PADDLEOCR/docs/version3.x/pipeline_usage/PP-StructureV3.en.md#L3082-L3450) và [code wrapper](../references/PADDLEOCR/paddleocr/_pipelines/pp_structurev3.py).

## 5. POST /restructure-pages — PaddleOCR-VL

[Route HPS](../references/PADDLEOCR/deploy/paddleocr_vl_docker/hps/gateway/app.py#L344-L356); [schema](../references/PADDLEOCR/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md#L2590-L2705); [wrapper](../references/PADDLEOCR/paddleocr/_pipelines/paddleocr_vl.py#L218-L229). Mục đích: xử lý liên trang, chẳng hạn gộp bảng và đánh lại cấp tiêu đề, không chạy lại OCR.

Request bắt buộc `pages` là array; mỗi phần tử lấy `prunedResult` từ infer và `markdownImages` từ `markdown.images` (object/null). Tùy chọn: `mergeTables`, `relevelTitles`, `concatenatePages`, `prettifyMarkdown` (mặc định true), `showFormulaNumber` (false), `returnMarkdownImages` (true), `outputFormats` (chỉ `docx`). Wrapper Python mặc định `merge_tables=true`, `relevel_titles=true`, `concatenate_pages=false`; không coi đó là bảo đảm cho mọi phiên bản HTTP backend.

```json
{
  "pages": [
    {"prunedResult": "<object prunedResult từ trang 1>", "markdownImages": null},
    {"prunedResult": "<object prunedResult từ trang 2>", "markdownImages": null}
  ],
  "mergeTables": true,
  "relevelTitles": true
}
```

Các chuỗi placeholder ở trên phải được thay bằng object thực. Response `result.layoutParsingResults` chứa các kết quả sau tái cấu trúc, theo schema infer nhưng không có ảnh trực quan/trung gian. `concatenatePages` có thể làm thay đổi cách nhóm kết quả.

## 6. Health và vận hành HPS

- `GET /health`: kiểm tra tiến trình gateway còn sống, HTTP 200, `{"logId":"…","errorCode":0,"errorMsg":"Healthy"}`; không kiểm tra dependencies.
- `GET /health/ready`: kiểm tra Triton server, cả model `layout-parsing` và `restructure-pages`, sau đó gọi `/health` của VLM. HTTP 200 với `errorMsg: "Ready"`; HTTP 503 nếu chưa sẵn sàng/timeout/lỗi. [Code](../references/PADDLEOCR/deploy/paddleocr_vl_docker/hps/gateway/app.py#L137-L214).

HPS mặc định giới hạn đồng thời 16 inference và 64 non-inference, timeout 600 giây, health timeout 5 giây; semaphore nằm trong state của từng process, không phải quota toàn cục giữa workers. Xem [cấu hình](../references/PADDLEOCR/deploy/paddleocr_vl_docker/hps/gateway/app.py#L34-L52) và [README](../references/PADDLEOCR/deploy/paddleocr_vl_docker/hps/README_en.md#L102-L125). Code gateway không định nghĩa xác thực; cần bảo vệ service và kiểm soát URL đầu vào ở tầng triển khai, đặc biệt rủi ro SSRF, kích thước file và dữ liệu nhạy cảm.

## 7. API của các pipeline khác

### OCR: POST /ocr

Request bắt buộc `file`; `fileType` và `visualize` tương tự mục 3–4. Các tùy chọn gồm `useDocOrientationClassify`, `useDocUnwarping`, `useTextlineOrientation`, `textDetLimitSideLen`, `textDetLimitType`, `textDetThresh`, `textDetBoxThresh`, `textDetUnclipRatio`, `textRecScoreThresh`. Response `result.ocrResults` là tập kết quả theo trang, mỗi phần tử có `prunedResult`, `ocrImage`, `docPreprocessingImage`, `inputImage`; `result.dataInfo` mô tả đầu vào. Ảnh có thể null và là Base64 hoặc URL tùy cấu hình. Nguồn: [HTTP schema OCR](../references/PADDLEOCR/docs/version3.x/pipeline_usage/OCR.en.md#L1972-L2190), [code wrapper OCR](../references/PADDLEOCR/paddleocr/_pipelines/ocr.py).

### PP-ChatOCRv4

Các API này thuộc service PP-ChatOCRv4, thường dùng theo luồng visual → vector tùy chọn → chat; MLLM là nhánh tùy chọn:

| POST | Input bắt buộc | Output trong `result` | Vai trò |
|---|---|---|---|
| `/chatocr-visual` | `file` | `layoutParsingResults`, `visualInfo`, `dataInfo` | Phân tích ảnh/PDF |
| `/chatocr-vector` | `visualInfo` array | `vectorInfo` object | Xây vector store; tùy chọn `minCharacters`, `blockSize`, `retrieverConfig` |
| `/chatocr-mllm` | `image` URL/Base64, `keyList` array | `mllmPredictInfo` object | Gọi MLLM; tùy chọn `mllmChatBotConfig` |
| `/chatocr-chat` | `keyList` array, `visualInfo` | `chatResult` object | Trích xuất thông tin; có thể dùng `vectorInfo`, `mllmPredictInfo`, cấu hình chatbot/retriever |

Tài liệu upstream mô tả `visualInfo` output của visual là array nhưng bảng input chat ghi object; cần kiểm tra phiên bản/schema service đang chạy thay vì tự giả định chuyển đổi. Cấu hình LLM/retriever có thể cần service và credentials riêng. [Schema HTTP](../references/PADDLEOCR/docs/version3.x/pipeline_usage/PP-ChatOCRv4.en.md#L2864-L3320); [code wrapper](../references/PADDLEOCR/paddleocr/_pipelines/pp_chatocrv4_doc.py).

### PP-DocTranslation

- `POST /doctrans-visual`: bắt buộc `file`, nhận `fileType`, các tùy chọn layout/OCR/table, `visualize`, `outputFormats`. Trả `result.layoutParsingResults` chứa `prunedResult`, `markdown`, ảnh và exports tùy chọn, cùng `dataInfo`.
- `POST /doctrans-translate`: bắt buộc `markdownList` array lấy từ kết quả visual; tùy chọn `targetLanguage`, `chunkSize`, `taskDescription`, `outputFormat`, `rulesStr`, `fewShotDemoTextContent`, `fewShotDemoKeyValueList`, `glossary`, `llmRequestInterval`, `chatBotConfig`. Trả `result.translationResults` array, mỗi phần tử gồm `language` và `markdown`. Có thể cần LLM/config riêng.

Nguồn: [schema HTTP PP-DocTranslation](../references/PADDLEOCR/docs/version3.x/pipeline_usage/PP-DocTranslation.en.md#L2785-L3200), [code wrapper](../references/PADDLEOCR/paddleocr/_pipelines/pp_doctranslation.py).
