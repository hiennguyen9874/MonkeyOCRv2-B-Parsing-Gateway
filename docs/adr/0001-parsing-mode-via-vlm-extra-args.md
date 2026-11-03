# Chọn chế độ phân tích qua `vlmExtraArgs.parsingMode`

Client chọn Chế độ hai giai đoạn hoặc Chế độ end-to-end theo từng request bằng `vlmExtraArgs.parsingMode` (`"two-stage"` | `"end-to-end"`). Nếu request bỏ trống, giá trị mặc định lấy từ cấu hình server. Cách này giữ request hợp lệ theo schema PaddleOCR-VL, nên client cũ như WeKnora không cần sửa, và một instance phục vụ được cả hai chế độ.

## Considered Options

- Chỉ cấu hình phía server: đơn giản, nhưng muốn đổi chế độ thì phải chạy hai instance.
- `useLayoutDetection=false` → end-to-end: bị loại vì ở PaddleOCR-VL giá trị này nghĩa là không có layout và cả trang là một Khối, trong khi end-to-end vẫn trả về layout.
