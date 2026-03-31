# 🚀 Workflow — Enterprise R&D Copilot (Cross-lingual RAG + RAPTOR)

## 🎯 Goal

Hệ thống trả lời câu hỏi Tiếng Việt dựa trên tài liệu kỹ thuật Tiếng Anh (PDF), đảm bảo:

- Factual correctness (không hallucination)
- Low latency
- Scalable production deployment

---

# 🧩 OVERALL FLOW

User Query (Vietnamese)
        ↓
[Agent 1: RAG-Router]
        ↓
Translate → Multi-query Expansion (EN)
        ↓
Embedding + Hybrid Search (Qdrant)
        ↓
Reranking (Python microservice)
        ↓
Top-K Context (Leaf + Parent nodes)
        ↓
[Agent 2: Analyst + Self-check]
        ↓
(Conditional)
[Agent 3: Reviewer]
        ↓
Final Answer (Vietnamese + citations)

---

# 🔄 STAGE 1 — INDEXING PIPELINE (OFFLINE)

## 1.1 Data Ingestion

- Source: ArXiv API
- Input:
  - PDF
  - Metadata (title, authors, year, abstract)

---

## 1.2 PDF Processing

- Extract raw text from PDF
- Clean:
  - remove noise (headers, footers, page numbers)
  - remove LaTeX artifacts
  - normalize encoding (UTF-8)

---

## 1.3 Semantic Chunking (Leaf Nodes)

- Split theo semantic boundaries (NOT fixed size)
- Preserve:
  - algorithm description
  - tables
  - formulas

---

## 1.4 Context Injection

Format mỗi chunk:

```
[Paper Title (Year)] Section Name:
<chunk content>
```

Ví dụ:

```
[Attention Is All You Need (2017)] Multi-Head Attention:
Multi-head attention allows the model to jointly attend to information
from different representation subspaces at different positions...
```

---

## 1.5 Embedding

- Model: bge-m3
- Output:
  - dense vector
  - metadata

---

## 1.6 Clustering (RAPTOR)

- Reduce dimension: UMAP
- Clustering: GMM

---

## 1.7 Parent Node Generation

- Input: cluster of leaf nodes
- LLM summarize → parent node (EN)

---

## 1.8 Storage (Qdrant)

Store BOTH:

### Leaf nodes

- raw chunk
- embedding
- metadata

### Parent nodes

- summary
- embedding
- reference to children

---

# 🔍 STAGE 2 — QUERY PROCESSING (ONLINE)

## 2.1 Input

- User query (Vietnamese)

---

## 2.2 Translate-first Strategy

- LLM translate → English
- Generate 3 variants:
  - semantic rephrase
  - keyword-focused
  - technical phrasing

---

## 2.3 Terminology Injection (Optional but recommended)

- Preserve keywords:
  - RAG
  - Transformer
  - LoRA
  - RLHF

---

# 🔎 STAGE 3 — RETRIEVAL

## 3.1 Embedding

- Embed 3 English queries

---

## 3.2 Hybrid Search

- Dense search (vector similarity)
- Sparse search (BM25)

Run in parallel

---

## 3.3 Merge Results

- Combine dense + sparse results
- Deduplicate

---

## 3.4 Reranking (Python microservice)

- Model: bge-reranker-v2-m3
- Input:
  - EN query
  - candidate chunks
- Output:
  - relevance score

---

## 3.5 Context Selection

- Select Top-K (e.g., 5–8)
- Mix:
  - Leaf nodes (detail)
  - Parent nodes (summary)

---

# 🧠 STAGE 4 — GENERATION

## 4.1 Agent 2: Analyst + Self-check

### Input

- Top-K context (EN)
- Original query (VN)

### Output

- Answer (Vietnamese)

---

## 4.2 Constraints (VERY IMPORTANT)

- Only use provided context
- Keep technical terms in English
- Must include citation:

```
Theo [Paper Title] (Author, Year), ...
```

Ví dụ:

```
Theo [Attention Is All You Need] (Vaswani et al., 2017),
cơ chế multi-head attention cho phép model jointly attend
tới thông tin từ các representation subspace khác nhau.
```

---

## 4.3 Self-check step

LLM must verify:

- numbers (accuracy, F1, params)
- terminology correctness
- consistency with context

---

# ⚖️ STAGE 5 — CONDITIONAL REVIEW

## Trigger conditions

- Query asks:
  - numbers / metrics
  - formulas
  - cost / performance
- OR low confidence from Analyst

---

## Reviewer Agent

- Compare:
  - Answer (VN)
  - Source (EN)
- Detect:
  - hallucination
  - incorrect translation
  - wrong numbers

---

## Output

- Approve OR regenerate answer

---

# 📊 STAGE 6 — EVALUATION (OFFLINE)

## Tools

- Ragas
- DeepEval

---

## Metrics

### 1. Context Recall

- Retrieved đúng tài liệu không?

### 2. Faithfulness

- Answer có đúng với source không?

### 3. Hallucination Rate

- Có bịa số liệu không?

---

# ⚡ PERFORMANCE OPTIMIZATION

## 1. Async Processing (Rust)

- Parallel:
  - translation
  - retrieval
  - BM25 + vector search

---

## 2. Caching

- Query → embedding
- Query → results
- Rerank results

---

## 3. Conditional Execution

- Reviewer only when needed

---

# 🚨 FAILURE HANDLING

## Case: No relevant context

→ Return:

```
Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu.
Vui lòng thử hỏi lại với câu hỏi cụ thể hơn hoặc về chủ đề AI/ML khác.
```

---

## Case: Low confidence

→ Trigger reviewer OR fallback answer

---

# 🧱 TECH STACK

- Backend: Rust (Actix-Web)
- Vector DB: Qdrant
- Embedding: bge-m3
- Reranker: bge-reranker-v2-m3
- LLM: Groq
- Evaluation: Ragas, DeepEval

---

# 🧠 DESIGN PRINCIPLES

- Translate-first > Cross-lingual embedding
- Raw data > Summary (for correctness)
- Conditional compute > Always-on agents
- Retrieval quality > Model size
