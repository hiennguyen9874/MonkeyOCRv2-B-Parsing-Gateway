# Chấp nhận và bỏ qua (no-op) các cờ PaddleOCR-VL không hỗ trợ

WeKnora luôn gửi `useSealRecognition=true`, `restructurePages/mergeTables/relevelTitles=true`, `promptLabel`, `layoutNms`, cùng `minPixels/maxPixels` đã tinh chỉnh cho Paddle. Nếu gateway từ chối các cờ này thì WeKnora không dùng được. Vì vậy gateway chấp nhận các cờ không có tương đương trong MonkeyOCRv2, không áp dụng chúng, và ghi hành vi thực tế vào `prunedResult.model_settings`. `minPixels/maxPixels` trong request bị bỏ qua; giới hạn pixel lấy theo mặc định upstream và cấu hình qua env, vì giá trị dành cho Paddle không phù hợp với độ phân giải MonkeyOCRv2 được huấn luyện.

Ngoại lệ là các cờ làm đổi hệ tọa độ của kết quả, như tiền xử lý (xem ADR-0004). Các cờ này bị từ chối thay vì bỏ qua.
