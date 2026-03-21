Workflow Dự Án: Hệ thống Agentic RAG Pháp lý Đa tác nhân (Tích hợp HyDE & RAPTOR) với Backend Rust
1. Tổng quan Kiến trúc (Architecture Overview)
Hệ thống là sự kết hợp giữa kiến trúc truy xuất nâng cao (Advanced RAG) và luồng suy luận Đa tác nhân (Multi-Agent), sử dụng sức mạnh của Groq API (miễn phí, tốc độ cao) và được tối ưu hóa toàn diện bằng backend viết bằng ngôn ngữ Rust để đạt độ trễ thấp nhất.

2. Giai đoạn 1: Chuẩn bị Dữ liệu & Lập chỉ mục Phân cấp (RAPTOR)
Giai đoạn này giúp hệ thống hiểu được cả bức tranh toàn cảnh lẫn chi tiết của các bộ luật dài.

Nguồn Dữ liệu: Sử dụng bộ dữ liệu undertheseanlp/UTS_VLC chứa các văn bản luật và nghị định Việt Nam.

Xử lý văn bản tốc độ cao: Dùng thư viện PyO3 kết hợp Rust vào Python để tăng tốc quá trình làm sạch (cleaning) và phân mảnh (chunking) tài liệu.

Xây dựng cây RAPTOR:

Chia nhỏ văn bản luật thành các đoạn (chunks) cơ sở.

Nhúng (Embed) các đoạn này bằng mô hình mã nguồn mở BAAI/bge-m3.

Dùng thuật toán UMAP và GMM để phân cụm (clustering) các điều khoản có ý nghĩa liên quan.

Gọi Groq API (LLaMA-3) tóm tắt từng cụm, tiếp tục lặp lại đệ quy để tạo thành một cây tri thức từ dưới lên trên.

Lưu trữ: Đưa toàn bộ các nút (nodes) của cây RAPTOR vào Qdrant (Vector Database viết bằng Rust).

3. Giai đoạn 2: Biến đổi Truy vấn với HyDE (Query Transformation)
Đây là tuyến phòng thủ đầu tiên khi nhận câu hỏi từ người dùng nhằm mở rộng ngữ cảnh.

Sinh tài liệu giả định: Ngay khi nhận câu hỏi, LLM sẽ sinh ra một câu trả lời nháp (có thể chứa thông tin chưa chuẩn xác nhưng cấu trúc và từ vựng mang đậm tính pháp lý).

Mã hóa (Embedding): Chuyển đổi câu trả lời giả định này thành vector để chuẩn bị cho quá trình tìm kiếm.

4. Giai đoạn 3: Truy xuất Lai & Xếp hạng lại (Hybrid Search & Reranking)
Hybrid Search: Đối chiếu vector của HyDE vào Qdrant kết hợp tìm kiếm ngữ nghĩa (Dense Vector) và tìm kiếm từ khóa chính xác (BM25 - Sparse Vector) để đảm bảo không trượt các số hiệu luật cụ thể.

Reranking: Top 20 kết quả thu về sẽ đi qua một mô hình Cross-encoder chạy cục bộ (như bge-reranker) để chấm điểm chéo từng tài liệu với câu hỏi gốc, giữ lại đúng 5 đoạn ngữ cảnh giá trị nhất.

5. Giai đoạn 4: Điều phối Đa tác nhân (Agentic Workflow) bằng Rust
Thay vì dùng Python FastAPI thông thường, backend được kiến trúc hoàn toàn bằng Rust để thể hiện năng lực kỹ sư hệ thống.

Backend Framework: Sử dụng Actix-Web để xử lý API đồng thời với hiệu năng cực cao và an toàn bộ nhớ.

AI SDK: Sử dụng framework rig-rs (Rust) để thiết lập luồng giao tiếp với Groq API và Qdrant.

Luồng Đa tác nhân (Lấy cảm hứng từ hệ thống Compliance):

Router Agent: Nhận câu hỏi, đánh giá ý định để quyết định xem có cần tra cứu luật hay chỉ là giao tiếp thông thường.

RAG Agent: Thực thi toàn bộ Giai đoạn 2 và 3 (HyDE + Hybrid Search + Reranking) để lấy bằng chứng pháp lý.

Analyst Agent: Đọc bằng chứng từ RAG Agent và sinh ra câu trả lời lập luận từng bước.

Consistency/Compliance Agent: Đóng vai trò "Thẩm phán". Tác nhân này đối chiếu chéo câu trả lời của Analyst Agent với các đoạn luật gốc để bắt lỗi logic hoặc ảo giác (hallucination). Nếu phát hiện lỗi, nó yêu cầu sinh lại câu trả lời.

6. Giai đoạn 5: Đánh giá Tự động bằng Dữ liệu Tổng hợp (Synthetic Data Evaluation)
Để chứng minh dự án đạt chuẩn "Production-ready", cần có hệ thống đo lường minh bạch.

Sử dụng framework ragas hoặc deepeval kết hợp LLM để tự động quét qua bộ luật và sinh ra hàng trăm cặp "câu hỏi khó - đáp án chuẩn" (Synthetic Data).

Chạy hệ thống RAG qua bộ dữ liệu này để lấy các chỉ số đo lường: Context Precision (Độ chính xác ngữ cảnh) và Faithfulness (Độ trung thực). Ghi chép các chỉ số này vào CV.