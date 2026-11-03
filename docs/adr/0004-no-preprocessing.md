# Không hỗ trợ tiền xử lý tài liệu; từ chối khi request bật

Tiền xử lý của MonkeyOCRv2 (chỉnh hướng, dewarp) cần torch/GPU và làm bbox chuyển sang hệ tọa độ của ảnh đã xử lý. Gateway không hỗ trợ tiền xử lý: `useDocOrientationClassify` và `useDocUnwarping` nhận `null` hoặc `false`; giá trị `true` trả về lỗi rõ ràng thay vì bị bỏ qua, vì bỏ qua sẽ trả tọa độ khác với điều client yêu cầu. Lựa chọn này khớp với run hai giai đoạn đã kiểm chứng trong legal-vi-ocr, vốn cũng tắt tiền xử lý.
