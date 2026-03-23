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

1. Giai đoạn 2: Biến đổi Truy vấn với Multi-Query Expansion (Query Transformation)
Đây là tuyến phòng thủ đầu tiên khi nhận câu hỏi từ người dùng nhằm mở rộng ngữ cảnh, thay thế hoàn toàn cho thuật toán HyDE để tránh bị "ngộ độc vector" do LLM ảo giác ra số hiệu luật.

Mở rộng Đa truy vấn (Multi-Query): Hệ thống dùng LLM (Groq) để phân tích câu hỏi gốc và sinh ra 3 biến thể câu hỏi khác nhau nhằm tăng cơ hội vét cạn (recall) tài liệu.

Mã hóa (Embedding): Chuyển đổi tất cả các biến thể truy vấn này thành vector để chuẩn bị tìm kiếm.

1. Giai đoạn 3: Truy xuất Lai & Xếp hạng lại (Hybrid Search & Reranking)
Hybrid Search: Tìm kiếm trên Qdrant kết hợp tìm ngữ nghĩa (Dense Vector) và tìm từ khóa chính xác (BM25 - Sparse Vector có áp dụng Log-scaled TF và loại bỏ Stop-words tiếng Việt). Hợp nhất kết quả bằng thuật toán RRF.

Reranking: Gọi hệ thống Microservice Python bằng HTTP để chạy mô hình Cross-encoder cục bộ (bge-reranker-v2-m3). Tốc độ siêu tốc thông qua batch-processing, chấm điểm chéo từng tài liệu để lấy lại chính xác top 5 đoạn ngữ cảnh giá trị nhất.

1. Giai đoạn 4: Điều phối Đa tác nhân (Agentic Workflow) bằng Rust
Thay vì dùng Python FastAPI thông thường, backend được kiến trúc hoàn toàn bằng Rust để thể hiện năng lực kỹ sư hệ thống.

Backend Framework: Sử dụng Actix-Web để xử lý API đồng thời với hiệu năng cực cao và an toàn bộ nhớ.

AI SDK: Sử dụng framework rig-rs (Rust) để thiết lập luồng giao tiếp với Groq API và Qdrant.

Luồng Đa tác nhân:

Router Agent: Nhận câu hỏi, đánh giá ý định để quyết định xem có cần tra cứu luật hay chỉ là giao tiếp thông thường.

RAG Agent: Thực thi toàn bộ Giai đoạn 2 và 3 (Multi-Query Expansion + Hybrid Search + Reranking) để lấy bằng chứng pháp lý.

Analyst Agent: Đọc bằng chứng từ RAG Agent và sinh ra câu trả lời lập luận từng bước.

Consistency/Compliance Agent: Đóng vai trò "Thẩm phán". Tác nhân này đối chiếu chéo câu trả lời của Analyst Agent với các đoạn luật gốc để bắt lỗi logic hoặc ảo giác (hallucination). Nếu phát hiện lỗi, nó yêu cầu sinh lại câu trả lời.

1. Giai đoạn 5: Đánh giá Tự động bằng Dữ liệu Tổng hợp (Synthetic Data Evaluation)
Để chứng minh dự án đạt chuẩn "Production-ready", cần có hệ thống đo lường minh bạch bằng tập test tự sinh.

Sử dụng script Python kết hợp LLM để tự động quét qua bộ luật và sinh ra hàng chục cặp "câu hỏi khó - đáp án chuẩn" (Synthetic Data).

Chạy hệ thống RAG qua bộ dữ liệu này và đánh giá tự động dựa trên 6 metric. Quan trọng nhất là: Context Recall (Độ đầy đủ của thuật toán tìm kiếm), Hallucination Rate (Tỉ lệ trích dẫn sai luật), và Faithfulness (Độ trung thực).
