# Rules cho BTVN#3 (ưu tiên từ cao xuống thấp)

Áp dụng cho toàn bộ mã, test và báo cáo trong thư mục này. Khi hai rule xung đột, rule có mức ưu tiên cao hơn thắng.

## P0 — An toàn và tính đúng đắn

1. Mọi chuyến bay, đặt chỗ, thanh toán và phê duyệt đều là dữ liệu **mock**. Không gọi API đặt vé/thanh toán thật; không lưu thẻ, khóa API hoặc bí mật trong repo.
2. Model chỉ **đề xuất** hành động. Harness phải kiểm tên tool, schema tham số, ràng buộc chuyến bay và quyền hạn **trước** khi chạy tool có tác dụng phụ.
3. `book_seat`/`pay` phải cần phê duyệt minh thị theo policy; khi chưa được duyệt, dừng ở trạng thái chờ, ghi rõ hành động định làm và lý do. Chống tác dụng phụ trùng khi retry.
4. Chỉ báo thành công khi code đọc lại booking và xác nhận `confirmed`, `paid`, đúng tuyến/ngày/giờ/giá và dữ liệu khớp với kết quả tool. Văn bản từ model không phải bằng chứng hoàn thành.

## P1 — Harness có giới hạn và giải thích được

5. Có giới hạn hữu hạn cho số lượt model, số tool call, thời gian; nếu lấy được usage thì ghi thêm token. Hết giới hạn phải trả trạng thái dở dang và handoff, không báo thành công.
6. Dùng observation JSON có trạng thái rõ (`ok`, `empty`, `invalid_param`, `error`, `denied`); không diễn dịch `{}` hoặc timeout thành “không có chuyến”.
7. Ghi trace có thứ tự: model đề xuất gì, tool nào được/không được chạy, observation, tiến triển, lý do dừng. Che bí mật nếu có.
8. Phát hiện gọi `(tool, args)` lặp vô ích và bế tắc không tiến triển; cho phép polling `get_booking` có tiến triển trạng thái.

## P2 — Ba mẫu thiết kế và đánh giá công bằng

9. Cài ReAct, Plan-then-Execute và mẫu Lai bằng LangChain/LangGraph; cả ba dùng **cùng** mock tools, policy, tiêu chí hoàn thành và giới hạn.
10. Plan-then-Execute lập kế hoạch nhìn được trước khi chạy; mẫu Lai lập lại kế hoạch khi observation làm kế hoạch cũ không còn hợp lệ. ReAct chọn bước sau từng observation.
11. Đánh giá hiệu quả ba mẫu bằng cùng một LLM thật, cùng bộ kịch bản và dữ liệu mock cố định; báo riêng tỉ lệ thành công, lỗi ràng buộc, số lượt model/tool, token, độ trễ và số lần cần bàn giao. Model giả lập chỉ dùng cho test logic, không làm bằng chứng so sánh hiệu quả LLM.

## P3 — Chất lượng bàn giao

12. Nộp mã `.py`, hướng dẫn chạy, test và báo cáo tiếng Việt. Mọi số liệu trong báo cáo phải có lệnh tái lập hoặc được ghi rõ là chưa đo.
13. Viết test theo `docs/PLAN.md` trước/sát với implementation; chạy test offline trước khi kết luận hoàn thành. Dùng cấu hình qua biến môi trường, không hard-code model/key.
