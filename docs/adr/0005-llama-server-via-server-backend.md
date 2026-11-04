# Hỗ trợ llama-server qua `MOCR2_SERVER_BACKEND`

Gateway chạy được với hai Máy chủ suy luận: vLLM chạy model gốc, và `llama-server` của llama.cpp chạy bản GGUF `aenawi/MonkeyOCRv2-B-GGUF`. Biến môi trường `MOCR2_SERVER_BACKEND` (`vllm` mặc định, hoặc `llama-cpp`; giá trị khác là lỗi khi khởi động) chọn máy chủ cho cả tiến trình. Không có ghi đè theo request, vì máy chủ là thuộc tính của triển khai. Gateway không tự dò backend. Payload chỉ khác nhau ở những điểm mà llama-server cần khác:

- Với `llama-cpp`, gateway gửi system line `You are a helpful assistant.`. Model card GGUF yêu cầu dòng này, và đường vLLM offline của Upstream (`build_vllm_prompt`) cũng gửi nó. Với `vllm`, gateway giữ parity với đường server của Upstream (`_chat_completion`), tức là không gửi system line.
- Với `llama-cpp`, `repetitionPenalty` được gửi thành `repeat_penalty`; gateway không gửi `repeat_last_n`, để llama-server dùng cấu hình của nó. llama-server bỏ qua `repetition_penalty` mà không báo lỗi. Ở commit `ca8daad19`, mặc định cửa sổ là 64 token và 0 tắt penalty, nên không đảm bảo cùng ngữ nghĩa toàn bộ chuỗi như vLLM. Trước đây gateway gửi `repeat_last_n=-1`, nhưng commit này từ chối giá trị âm; xem issue #17. Phiên bản của triển khai báo lỗi chưa được xác định.

Readiness không đổi giữa hai máy chủ (`/health` và tên model có trong `/v1/models`). llama-server cần chạy với `--alias` trùng `MOCR2_SERVED_MODEL_NAME`.

## Consequences

Bản GGUF là Q8_0, có bộ mã hóa ảnh đã chuyển đổi và chạy trên engine khác, nên output không khớp từng ký tự với vLLM bf16. Không có test parity tự động cho mức độ lệch này. Script so sánh thủ công đo CER/diff giữa hai máy chủ trên trang công khai và lưu kết quả vào `docs/validation/llama-cpp/`. Kết quả này không có ngưỡng cứng.
