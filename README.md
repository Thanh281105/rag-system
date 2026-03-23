# 🏛️ Hệ thống RAG Pháp lý Đa tác nhân (Luật Việt Nam)

> Hệ thống Hệ thống truy xuất và trả lời câu hỏi pháp lý thông minh được thiết kế đặc thù cho cấu trúc Luật Việt Nam. Kết hợp kiến trúc **Multi-Agent + RAPTOR** với backend Rust hiệu năng siêu cao.

## ✨ Tính năng nổi bật & Tối ưu hóa

- **🌳 RAPTOR Tree Indexing (Hierarchy-Aware)** — Xây dựng cây tri thức phân cấp sâu tới tận cấp `Khoản/Điểm`, tự động *nhúng ngữ cảnh của Điều cha* vào từng đoạn cắt để LLM không bị mất dấu nguồn gốc văn bản.
- **📝 Multi-Query Expansion** — Rẽ nhánh đa câu hỏi từ truy vấn gốc (khắc phục điểm yếu "ảo giác" của thuật toán HyDE truyền thống).
- **🔍 Hybrid Search với Vietnamese BM25** — Kết hợp Dense (semantic) + Sparse search. Vector Sparse được loại bỏ Stop-words Tiếng Việt và áp dụng `Log-scaled TF` để bắt chính xác số hiệu luật (VD: "Nghị định 100", "Điều 345").
- **⚡ Fast Cross-Encoder Reranking** — Reranking top kết quả bằng mô hình `bge-reranker-v2-m3` qua Python Microservice thay vì dùng LLM-scoring chậm chạp, đem lại tốc độ siêu tốc cho Rust backend.
- **🤖 Multi-Agent Orchestration** — 4 tác nhân AI phối hợp: Router → RAG → Analyst → Compliance. Agent `Compliance` (Thẩm phán) có khả năng tự bắt lỗi ảo giác và yêu cầu sinh lại đáp án.
- **🦀 Rust Backend** — Code bởi Actix-Web + rig-rs với cơ chế xử lý song song (tokio) giúp giảm thiểu độ trễ.
- **📊 Auto Evaluation Chuyên Sâu** — Đánh giá tự động bao gồm các metrics tối quan trọng cho pháp lý: Context Recall, Hallucination Rate, Answer Correctness, Faithfulness.

## 🏗️ Kiến trúc

```
User Query
    ↓
[Router Agent] → Phân loại ý định (Casual vs Legal)
    ↓
[RAG Agent] → Multi-Query Expansion → Hybrid Search (Dense+Sparse) → Cross-Encoder Reranking
    ↓
[Analyst Agent] → Sinh câu trả lời lập luận dựa trên bằng chứng
    ↓
[Compliance Agent] → Kiểm tra ảo giác (Hallucination) / Trích dẫn sai (Retry nếu hỏng)
    ↓
Final Answer
```

## 🚀 Cài đặt

### Yêu cầu

- Python 3.10+
- Rust (rustup)
- Docker Desktop

### Bước 1: Clone & Setup

```bash
git clone <repo-url>
cd RAG_self_project
cp .env.example .env
# Điền GROQ_API_KEY và các setting (RRF_K, RRF_DENSE_WEIGHT...) vào .env
# Copy frontend `.env.example` -> `frontend/.env` nếu có
```

### Bước 2: Khởi động Qdrant

```bash
docker-compose up -d
```

### Bước 3: Cài đặt & Chạy Python Pipeline (Data + Microservices)

```bash
cd python
pip install -r requirements.txt

# Chạy tạo data và vector DB
python scripts/01_download_data.py
python scripts/02_build_raptor.py
python scripts/03_index_qdrant.py

# KHỞI ĐỘNG MICROSERVICE CHO RUST (Rất Quan Trọng)
python scripts/embedding_server.py
```

### Bước 4: Khởi động Rust Backend

```bash
# Mở một terminal mới
cd rust_backend
cargo run --release
```

## 📊 Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq API (LLaMA-3.3-70b/8b) |
| Embedding | BAAI/bge-m3 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Vector DB | Qdrant |
| Backend | Rust (Actix-Web + rig-rs) |
| Microservice | Python (FastAPI + SentenceTransformers) |
| Evaluation | Custom LLM-as-a-judge (Context Recall, Hallucination Rate) |

## 📁 Cấu trúc dự án

```
RAG_self_project/
├── python/              # Data processing (Cleaner, Chunker), RAPTOR, Evaluation, FastAPI Microservices
├── rust_backend/        # Multi-Agent Actix-Web server điều phối mọi hoạt động
├── frontend/            # Web UI
├── data/                # Raw/Processed/Synthetic data
└── docker-compose.yml   # Qdrant service
```

## 📜 License

MIT
