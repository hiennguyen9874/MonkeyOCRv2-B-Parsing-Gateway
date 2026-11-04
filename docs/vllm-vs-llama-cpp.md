# Triển khai với vLLM và llama.cpp

Tài liệu này tổng hợp [spec #12](https://github.com/hiennguyen9874/MonkeyOCRv2-B-Parsing-Gateway/issues/12), các ticket triển khai #13–#16 và các sửa lỗi [#17](https://github.com/hiennguyen9874/MonkeyOCRv2-B-Parsing-Gateway/issues/17), [#18](https://github.com/hiennguyen9874/MonkeyOCRv2-B-Parsing-Gateway/issues/18). Hướng dẫn chạy đầy đủ nằm trong [README](../README.md); các quyết định về payload nằm trong [ADR-0005](adr/0005-llama-server-via-server-backend.md).

## Tổng quan

Client như WeKnora vẫn gọi cùng API PaddleOCR-VL của gateway, đặc biệt là `POST /layout-parsing`; chuyển backend là thay đổi triển khai, không phải thay đổi client. Gateway vẫn xử lý ảnh và tài liệu, tạo prompt, phân tích output và dựng response. Máy chủ suy luận chạy riêng, cung cấp API tương thích OpenAI.

| Khía cạnh | vLLM | llama.cpp |
| --- | --- | --- |
| Chọn backend | `MOCR2_SERVER_BACKEND=vllm`, cũng là mặc định | `MOCR2_SERVER_BACKEND=llama-cpp` |
| Model | `zenosai/MonkeyOCRv2-B-Parsing` gốc | `aenawi/MonkeyOCRv2-B-GGUF`: text Q8_0 và vision projector F16 |
| Máy chủ | vLLM, chạy qua launcher Upstream để đăng ký model tùy biến | `llama-server` không sửa đổi, do operator tự build/cung cấp |
| Môi trường | Cài vLLM với một CUDA extra; hướng dẫn dùng GPU NVIDIA | Không cần vLLM extra cho gateway; build CUDA được khuyến nghị. Đã thử CPU cho kiểm tra #17, không cam kết hiệu năng CPU |
| System message | Không gửi, theo đường server của Upstream và chat template của model | Không gửi, giống vLLM (#18); chat template nhúng trong GGUF giống hệt bản gốc |
| `repetitionPenalty` | Gửi `repetition_penalty` | Gửi `repeat_penalty`, không gửi `repeat_last_n`; ngữ nghĩa cửa sổ có thể khác |
| Tên model | `MOCR2_SERVED_MODEL_NAME` khớp `--served-model-name` | `MOCR2_SERVED_MODEL_NAME` khớp `--alias` |

Backend được chọn cho **cả tiến trình**; không tự phát hiện và không đổi theo request. Giá trị không hợp lệ làm gateway lỗi khi khởi động. Hai backend đều hỗ trợ `two-stage` và `end-to-end`; mặc định là `two-stage`, có thể đổi bằng `MOCR2_PARSING_MODE` hoặc `vlmExtraArgs.parsingMode` theo request.

## Chuẩn bị và chạy

### vLLM

1. Khởi tạo submodule và tải model gốc vào `models/MonkeyOCRv2-B-Parsing`.
2. Cài một trong hai extra `vllm-cu126` hoặc `vllm-cu128`. Chạy `references/MonkeyOCRv2/parsing/serve.py` qua `uv run` với extra đã chọn; launcher đăng ký implementation model tùy biến. Đặt `--served-model-name MonkeyOCRv2`.
3. Chạy gateway với `MOCR2_SERVER_URL` trỏ tới server và `MOCR2_SERVED_MODEL_NAME=MonkeyOCRv2`. Có thể bỏ `MOCR2_SERVER_BACKEND` vì mặc định là `vllm`.

[README: Run locally with uv](../README.md#run-locally-with-uv) có lệnh đầy đủ, bao gồm CUDA, GPU, context và ví dụ gửi tài liệu.

### llama.cpp

1. Tự cung cấp `./llama.cpp/llama-server`; thư mục này và `models/` được gitignore, không được vendoring trong repo. Bản đã kiểm thử là commit `ca8daad19`, build 12072 với `GGML_CUDA=ON`; model card báo b11382 là build tối thiểu **đã thử**, không phải bảo đảm cho mọi build sau đó.
2. Tải hai file GGUF vào thư mục riêng:

   ```sh
   uv run --with huggingface-hub hf download aenawi/MonkeyOCRv2-B-GGUF \
     MonkeyOCRv2-B.q8_0.gguf MonkeyOCRv2-B.mmproj-f16.gguf \
     --local-dir models/MonkeyOCRv2-B-GGUF
   ```

3. Chạy server và gateway ở hai terminal từ thư mục gốc repo:

   ```sh
   CUDA_VISIBLE_DEVICES=0 ./llama.cpp/llama-server \
     -m models/MonkeyOCRv2-B-GGUF/MonkeyOCRv2-B.q8_0.gguf \
     --mmproj models/MonkeyOCRv2-B-GGUF/MonkeyOCRv2-B.mmproj-f16.gguf \
     -np 4 -c 65536 -ngl 99 --alias MonkeyOCRv2 --host 127.0.0.1 --port 19388
   ```

   ```sh
   MOCR2_SERVER_BACKEND=llama-cpp \
   MOCR2_SERVER_URL=http://127.0.0.1:19388 \
   MOCR2_SERVED_MODEL_NAME=MonkeyOCRv2 \
     uv run --locked uvicorn --factory monkeyocr_gateway.app:create_app \
     --host 127.0.0.1 --port 8080
   ```

   Cấu hình server này dùng 4 slot, tổng context 65.536 token (16.384 mỗi slot), và offload các layer lên GPU. Chọn GPU và port phù hợp; giữ loopback nếu chưa bảo vệ truy cập. Gateway không cần cài vLLM extra.

Với cả hai, kiểm tra `curl -i http://127.0.0.1:8080/health/ready` trước khi gửi traffic. Gateway yêu cầu server trả 200 tại `/health` **và** liệt kê đúng tên model tại `/v1/models`; sai tên hoặc model còn đang load sẽ not-ready. Với llama-server, kiểm tra `--alias` nếu tên không khớp.

## Khác biệt payload và hành vi

- Cả hai backend chỉ gửi user message (ảnh + prompt) trong mọi lời gọi: layout, nhận dạng khối, end-to-end và repeat-retry. Trước #18, gateway thêm system message `You are a helpful assistant.` cho llama.cpp theo model card GGUF. Dòng này không có trong template train (`default_system=None`), trong `chat_template.jinja` của model gốc, trong template nhúng trong GGUF, hay trong đường server chính của Upstream. Chỉ `build_vllm_prompt` trong biến thể CPU của Upstream có nó. Khi không có system message, llama-server render prompt giống vLLM, và token bao ảnh `<|img|>`/`<|endofimg|>` mang đúng id của `<|vision_start|>`/`<|vision_end|>`. Cần dùng template Jinja của GGUF (mặc định ở build 12072): không chạy với `--no-jinja` hay `--chat-template`. Xem [kiểm tra #18](validation/llama-cpp-system-message.md).
- Khi client không gửi `repetitionPenalty`, gateway không gửi trường penalty cho cả hai backend; server dùng mặc định của nó. Khi có, vLLM nhận `repetition_penalty`; llama-server nhận `repeat_penalty`. Gửi nguyên tên của vLLM cho llama-server có thể bị bỏ qua im lặng.
- Gateway **không** gửi `repeat_last_n` cho llama-server. Build 12072 mặc định xét 64 token gần nhất; giá trị 0 tắt penalty. Điều này không đảm bảo ngữ nghĩa toàn bộ chuỗi của vLLM. Không dùng quyết định cũ `repeat_last_n=-1` trong #13: #17 đã thay thế nó sau khi tái hiện HTTP 400 trên build 12072. Phiên bản và cấu hình của deployment báo lỗi ban đầu chưa được xác định; không suy rộng kết quả cho mọi phiên bản. Xem [kiểm tra và giới hạn #17](validation/llama-cpp-repeat-penalty.md).
- `temperature` (mặc định 0, greedy), `topP` → `top_p`, `maxNewTokens` → `max_tokens`, model, prompt và image data URI được xử lý giống nhau. Gateway vẫn resize ảnh theo giới hạn pixel như Upstream; không có nhánh tiền xử lý ảnh riêng cho llama.cpp.
- Readiness, timeout, retry, giới hạn đồng thời và ánh xạ lỗi 503/504/500 dùng chung. Thông báo gateway dùng tên trung lập “inference server”; chi tiết lỗi do server trả về vẫn có thể nhắc tên backend. Các biến vận hành được liệt kê trong [AGENTS.md](../AGENTS.md).

## Chất lượng và bằng chứng

Không kỳ vọng output khớp từng token: GGUF dùng quantization Q8_0, vision encoder đã chuyển đổi và engine khác với model gốc trên vLLM. Cùng temperature 0 không chứng minh hai backend cho cùng output hoặc cùng độ chính xác.

[So sánh thủ công #16](validation/llama-cpp.md) trên A100 dùng vLLM 0.11.2 và llama.cpp CUDA build 12072: hai ảnh tổng hợp, cả hai chế độ, có Markdown CER 0 và cùng số, nhãn, nội dung khối. Tuy nhiên, **cả hai request two-stage trên cả hai backend đều có output suy luận bị cắt do giới hạn token**; kết quả có thể không đầy đủ. Đây là phép đo độ lệch so với vLLM, không phải benchmark độ chính xác hay bằng chứng tương đương tổng quát. Không có ngưỡng pass/fail; script không chạy trong pytest. Báo cáo, response thô và lệnh tái hiện nằm trong [validation/llama-cpp](validation/llama-cpp/). Lần chạy đó còn gửi system message cho llama.cpp. [Chạy lại sau #18](validation/llama-cpp-system-message.md) với cùng cấu hình: lần trước có một bbox lệch 2 px, lần này cả label, nội dung và bbox của mọi khối đều khớp vLLM trong cả bốn trường hợp. Giới hạn về mẫu nhỏ và output bị cắt vẫn giữ nguyên.

Kiểm thử tự động dùng fake server để kiểm tra payload, response và xử lý lỗi; không thay thế chạy model thật. Khi đánh giá dữ liệu riêng, chỉ lưu fixture và kết quả công khai hoặc tổng hợp, không đưa dữ liệu nhạy cảm từ `legal-vi-ocr` vào repo.
