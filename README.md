# 🧠 Cross-lingual ArXiv RAG — Trợ lý Nghiên cứu AI

Hệ thống trả lời câu hỏi AI/ML bằng tiếng Việt, dựa trên papers ArXiv tiếng Anh.

```
Vietnamese Query → [Translate] → English Search → [RAPTOR + Hybrid] → Vietnamese Answer
```

## 🏗️ Architecture

```
User Query (VN)
      ↓
[Agent 1: RAG-Router]
├─ Classify intent (Technical/Casual)
├─ Translate VN → EN
├─ Multi-Query Expansion (3 EN variants)
├─ Hybrid Search (Dense + Sparse)
└─ Reranking (bge-reranker-v2-m3)
      ↓
[Agent 2: Analyst + Self-check]
├─ Generate VN answer from EN context
├─ Citation: Theo [Paper] (Author, Year)
└─ Self-verify numbers & terminology
      ↓
[Agent 3: Conditional Reviewer]
├─ Only triggers for: numbers, formulas, comparisons
├─ Compare VN answer vs EN source
└─ Approve OR regenerate (max 2 retries)
      ↓
Final Answer (Vietnamese + citations)
```

## 🧩 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Rust (Actix-Web) |
| **Orchestration** | 3-Agent Pipeline |
| **Vector DB** | Qdrant (Docker) |
| **Embedding** | BAAI/bge-m3 (1024D) |
| **Reranker** | bge-reranker-v2-m3 |
| **LLM** | Groq API (LLaMA-3.3-70b) |
| **Data Source** | ArXiv API (cs.AI) |
| **Indexing** | RAPTOR (UMAP + GMM) |
| **Evaluation** | Ragas + DeepEval (7 metrics) |

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Docker for Qdrant
docker-compose up -d

# Python dependencies
cd python && pip install -r requirements.txt
```

### 2. Environment Setup

```bash
cp .env.example .env
# Edit .env: add GROQ_API_KEY
```

### 3. Data Pipeline (Offline)

```bash
# Step 1: Download ArXiv papers
python python/scripts/01_download_data.py

# Step 2: Build RAPTOR tree
python python/scripts/02_build_raptor.py

# Step 3: Index to Qdrant
python python/scripts/03_index_qdrant.py
```

### 4. Start Backend

```bash
# Start Python embedding/reranking service
python python/scripts/embedding_server.py

# Start Rust backend
cd rust_backend
cargo run --release
```

### 5. Open UI

Visit `http://localhost:8080`

## 📊 Evaluation

```bash
# Generate synthetic Q&A (Vietnamese questions from English papers)
python python/scripts/04_evaluate.py
```

**7 Metrics:**

| Metric | Target |
|--------|--------|
| Faithfulness | ≥ 0.90 |
| Context Precision | ≥ 0.80 |
| Context Recall | ≥ 0.85 |
| Answer Relevancy | ≥ 0.85 |
| Answer Correctness | ≥ 0.80 |
| Hallucination Rate | ≤ 0.10 |
| Translation Faithfulness | ≥ 0.85 |

## 📁 Project Structure

```
├── python/
│   ├── data_processing/    # ArXiv loader, cleaner, chunker
│   ├── raptor/             # UMAP + GMM clustering, summarization
│   ├── retrieval/          # Qdrant client (hybrid search)
│   ├── evaluation/         # Synthetic data + 7 metrics
│   ├── embedding_server.py # FastAPI service (embed + rerank)
│   └── scripts/            # Pipeline scripts (01-04)
├── rust_backend/
│   ├── src/agents/         # rag.rs, analyst.rs, compliance.rs
│   ├── src/services/       # groq.rs, qdrant.rs, reranker.rs
│   └── src/routes/         # query.rs (3-agent orchestration)
├── frontend/               # Chat UI (HTML/CSS/JS)
├── workflow.md             # Detailed pipeline documentation
└── docker-compose.yml      # Qdrant service
```
