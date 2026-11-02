# Project Spec: Chatbot RAG Hỏi Đáp Quy Chế & Giáo Trình (Có Trích Nguồn)

## 1. Kiến trúc & Tech Stack Bắt buộc
- **Luồng xử lý:** PDF quy chế -> Chunk + Embed (BGE-M3) -> Qdrant Vector DB -> LLM đọc & trả lời -> Trả lời kèm trích dẫn chính xác số trang.
- **Tech Stack:** Python, FastAPI (Backend API), Qdrant (Vector DB), BAAI/bge-m3 (Embedding), Qwen/GPT/Gemini API (LLM), Streamlit (Frontend UI).
- **Tính năng ăn điểm (Key Requirements):**
  1. Giữ nguyên metadata `page_number` và `source_file` khi cắt chunk từ PDF để trích dẫn đúng số trang.
  2. Hỗ trợ Hybrid Search (kết hợp BM25 Sparse Vector + Dense Vector) và Rerank.
  3. Chống ảo giác (Anti-Hallucination): Nếu thông tin không có trong context truy xuất được từ Qdrant, bot bắt buộc trả lời "Không tìm thấy thông tin trong tài liệu quy chế", tuyệt đối không tự bịa.
  4. Bộ đánh giá tự động 50 câu hỏi (đo chỉ số Recall@5).

## 2. Quy trình Vibe Coding (Tiered Multi-Agent)
- **Tác vụ nhỏ:** Sửa lỗi vặt, giải thích code -> Trả lời trực tiếp.
- **Khi có từ khóa `[TEAM]` hoặc làm tính năng theo Tuần:**
  1. **[Lead Planner - Gemini 4 Argon]:** Chia nhỏ task, chọn chiến lược chunking tối ưu và định nghĩa schema dữ liệu.
  2. **[Builder - Subagent gemini-3.8-flash]:** Viết code hoàn chỉnh vào đúng các file trong thư mục `src/`, xử lý lỗi đầy đủ.
  3. **[Critic - Subagent gemini-4-argon]:** Kiểm tra code xem có làm mất `page_number` không, prompt chống bịa đã chặt chẽ chưa, và có lỗ hổng bảo mật/lộ API key không. Duyệt `APPROVED` hoặc bắt sửa lại `REJECTED`.
