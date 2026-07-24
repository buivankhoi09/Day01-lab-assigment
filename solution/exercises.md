# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Quy luật chung: temperature càng cao, độ "ngẫu nhiên/đa dạng" càng tăng, nhưng độ chính xác và tính nhất quán càng giảm

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Với chatbot hỗ trợ khách hàng, mình sẽ chọn temperature thấp, khoảng 0.2–0.3. Vì loại chatbot này cần trả lời chính xác, nhất quán, đúng chính sách công ty — không cần "sáng tạo", ngược lại sáng tạo quá dễ gây hiểu nhầm hoặc trả lời sai thông tin (hallucination), ảnh hưởng trải nghiệm và uy tín.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> - GPT-4o đắt hơn GPT-4o-mini khoảng 16.7 lần cho cùng số token output
-   Trường hợp đáng dùng GPT-4o: các tác vụ cần suy luận phức tạp, độ chính xác cao (phân tích hợp đồng, code review, tư vấn y tế/pháp lý). Trường hợp nên dùng mini: các tác vụ đơn giản, khối lượng lớn (trả lời FAQ, phân loại email, tóm tắt ngắn) — vì chênh lệch chất lượng không đáng để trả gấp 16 lần chi phí.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với persona "giáo viên tiểu học", phản hồi thường ngắn gọn hơn, dùng từ ngữ đơn giản, có ví dụ gần gũi đời thường (ví dụ so sánh blockchain với "cuốn sổ ghi chép chung mà ai cũng nhìn thấy"), tránh thuật ngữ chuyên môn. Với persona "chuyên gia tài chính", phản hồi dài hơn, dùng thuật ngữ kỹ thuật (hash, node phân tán, consensus mechanism...), đi sâu vào cơ chế hoạt động và ứng dụng thực tế. Điều này cho thấy system prompt định hình rất mạnh về giọng văn, độ sâu kiến thức, và đối tượng người đọc mà model hướng tới, dù câu hỏi user hoàn toàn giống nhau.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> - Số token đếm qua tiktoken là 165 token, trong khi ước lượng 100 / 0.75 chỉ khoảng 133 token (chênh lệch khoảng 24%)
    -tiếng Việt có dấu thanh và nhiều âm tiết đơn không thuộc bộ từ vựng phổ biến mà tokenizer (BPE) được huấn luyện chủ yếu trên tiếng Anh, nên 1 từ tiếng Việt thường bị tách thành nhiều token con hơn (đôi khi tách theo từng âm tiết hoặc cụm ký tự UTF-8), trong khi tiếng Anh có nhiều từ nguyên vẹn là 1 token.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng tương tác trực tiếp với người dùng như chatbot, trợ lý ảo — vì người dùng thấy phản hồi xuất hiện ngay lập tức, giảm cảm giác chờ đợi dù tổng thời gian xử lý không đổi, tạo trải nghiệm mượt mà tự nhiên như đang trò chuyện thật. Ngược lại, non-streaming phù hợp hơn khi hệ thống cần xử lý toàn bộ phản hồi trước khi dùng (ví dụ: parse JSON, gọi tiếp một hàm xử lý dữ liệu, hoặc chạy batch job không có người dùng trực tiếp theo dõi) — lúc đó stream từng phần không có ý nghĩa và chỉ làm code phức tạp hơn.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> So với delay cố định, exponential backoff giúp giãn cách các lần thử lại ra xa nhau dần, giảm áp lực dồn dập lên server đang quá tải, cho server có thời gian phục hồi. Nếu hàng nghìn client cùng retry với delay cố định giống nhau (ví dụ luôn chờ đúng 1 giây), tất cả sẽ gửi lại request cùng lúc theo từng đợt đồng bộ — gây ra hiện tượng "thundering herd" (đàn trâu cùng lao vào), khiến server vốn đã quá tải càng sập nặng hơn thay vì được giảm tải.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt."
Giải thích: yêu cầu "trả lời ngắn gọn" để tránh model lan man, tiết kiệm token/chi phí và giữ hội thoại dễ theo dõi trong khung chat. Chỉ định "bằng tiếng Việt" để đảm bảo nhất quán ngôn ngữ, tránh model tự chuyển sang tiếng Anh khi gặp thuật ngữ kỹ thuật.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất: history chỉ giữ 3 lượt gần nhất (6 message), nên trợ lý "quên" hoàn toàn ngữ cảnh các lượt trò chuyện trước đó — nếu người dùng hỏi lại thứ đã nói ở lượt 5-6 câu trước, model sẽ không nhớ. Đề xuất cải thiện: thêm một lớp tóm tắt (summarization) — khi history vượt quá giới hạn, dùng chính LLM tóm tắt các lượt cũ thành 1 đoạn ngắn rồi lưu vào 1 message system bổ sung, thay vì xoá hẳn — vừa giữ được ngữ cảnh dài hạn, vừa không làm phình token quá mức.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
