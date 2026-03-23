"""
Embedding & Reranking Microservice - Phục vụ cho Rust Backend.
Chạy song song với Rust server để cung cấp:
  1. Embedding thật từ BAAI/bge-m3 (endpoint /embed)
  2. Cross-encoder reranking từ BAAI/bge-reranker-v2-m3 (endpoint /rerank)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Biến global cho models
_embed_model = None
_rerank_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models khi server start."""
    global _embed_model, _rerank_model
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from config import EMBEDDING_MODEL, RERANKER_MODEL

    print(f"🔄 Đang tải mô hình embedding: {EMBEDDING_MODEL}...")
    _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✅ Mô hình embedding sẵn sàng ({_embed_model.get_sentence_embedding_dimension()}D)")

    print(f"🔄 Đang tải mô hình reranker: {RERANKER_MODEL}...")
    _rerank_model = CrossEncoder(RERANKER_MODEL, max_length=512)
    print(f"✅ Mô hình reranker sẵn sàng")

    yield
    print("🛑 Server đang tắt...")


app = FastAPI(title="Legal RAG Embedding & Reranking Service", lifespan=lifespan)


# ─── Embedding Endpoint ──────────────────────────────────

class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    vector: list[float]
    dimension: int


@app.post("/embed", response_model=EmbedResponse)
async def embed_text(req: EmbedRequest):
    """Embed một đoạn văn bản thành vector."""
    vector = _embed_model.encode(req.text, normalize_embeddings=True).tolist()
    return EmbedResponse(vector=vector, dimension=len(vector))


# ─── Reranking Endpoint ──────────────────────────────────

class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_k: int = 5


class RerankResult(BaseModel):
    index: int
    score: float


class RerankResponse(BaseModel):
    results: list[RerankResult]


@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(req: RerankRequest):
    """
    Rerank documents bằng Cross-encoder thật (bge-reranker-v2-m3).
    Trả về top_k documents đã sắp xếp theo điểm relevance giảm dần.
    """
    if not req.documents:
        return RerankResponse(results=[])

    # Tạo cặp (query, document) cho cross-encoder
    pairs = [(req.query, doc) for doc in req.documents]

    # Chấm điểm bằng cross-encoder
    scores = _rerank_model.predict(pairs)

    # Tạo danh sách (index, score) và sắp xếp giảm dần
    indexed_scores = [(i, float(s)) for i, s in enumerate(scores)]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    # Giữ top_k
    top_results = indexed_scores[:req.top_k]

    return RerankResponse(
        results=[RerankResult(index=idx, score=score) for idx, score in top_results]
    )


# ─── Health Check ────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "embed_model_loaded": _embed_model is not None,
        "rerank_model_loaded": _rerank_model is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
