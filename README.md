# 🏛️ Agentic RAG Pháp lý Đa tác nhân

> Hệ thống truy xuất và trả lời câu hỏi pháp lý thông minh, sử dụng kiến trúc **Multi-Agent + RAPTOR + HyDE** với backend Rust hiệu năng cao.

## ✨ Tính năng nổi bật

- **🌳 RAPTOR Tree Indexing** — Xây dựng cây tri thức phân cấp từ văn bản luật
- **📝 HyDE Query Transformation** — Mở rộng ngữ cảnh truy vấn bằng tài liệu giả định
- **🔍 Hybrid Search + Reranking** — Kết hợp Dense + BM25 Sparse search với Cross-encoder
- **🤖 Multi-Agent Orchestration** — 4 tác nhân AI phối hợp: Router → RAG → Analyst → Compliance
- **⚡ Rust Backend** — Actix-Web + rig-rs cho hiệu năng cực cao
- **📊 Auto Evaluation** — Đánh giá tự động bằng Synthetic Data (ragas/deepeval)

## 🏗️ Kiến trúc

```
User Query
    ↓
[Router Agent] → Phân loại ý định
    ↓
[RAG Agent] → HyDE → Hybrid Search → Reranking
    ↓
[Analyst Agent] → Sinh câu trả lời lập luận
    ↓
[Compliance Agent] → Kiểm tra hallucination
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
# Điền GROQ_API_KEY vào .env
```

### Bước 2: Khởi động Qdrant
```bash
docker-compose up -d
```

### Bước 3: Cài Python dependencies
```bash
cd python
pip install -r requirements.txt
```

### Bước 4: Chạy Pipeline
```bash
python scripts/01_download_data.py
python scripts/02_build_raptor.py
python scripts/03_index_qdrant.py
```

### Bước 5: Khởi động Rust Backend
```bash
cd rust_backend
cargo run --release
```

## 📊 Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq API (LLaMA-3.3-70b) |
| Embedding | BAAI/bge-m3 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Vector DB | Qdrant |
| Backend | Rust (Actix-Web + rig-rs) |
| Text Processing | Rust (PyO3) |
| Evaluation | ragas / deepeval |

## 📁 Cấu trúc dự án

```
RAG_self_project/
├── python/              # Data processing, RAPTOR, HyDE, Retrieval
├── rust_backend/        # Multi-Agent Actix-Web server
├── rust_text_processor/ # PyO3 text processing extension
├── frontend/            # Web UI
├── data/                # Raw/Processed/Synthetic data
└── docker-compose.yml   # Qdrant service
```

## 📜 License

MIT
