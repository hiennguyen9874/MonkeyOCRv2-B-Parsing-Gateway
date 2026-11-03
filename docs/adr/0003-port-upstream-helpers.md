# Port các helper thuần của upstream thay vì import `core_runner`

`references/MonkeyOCRv2/parsing/core_runner.py` import `torch` và `modeling.modeling_preprocessor` ngay ở đầu module. Gateway không chạy tiền xử lý và không cần GPU, nên không nên kéo theo torch. Gateway port các helper thuần (prompt, parse output có khả năng chịu lỗi, ánh xạ bbox, `otsl_to_html`, `process_formula`) từ revision đã pin `106914e`. Mỗi helper ghi rõ nguồn, và có test parity dùng lại test của upstream để phát hiện lệch hành vi khi nâng revision.

## Consequences

Khi nâng revision upstream, phải so sánh lại các helper đã port; test parity là cơ chế cảnh báo.
