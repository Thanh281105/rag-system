"""
Embedding Microservice - Phục vụ vector embedding cho Rust Backend.
Chạy song song với Rust server để cung cấp embedding thật từ BAAI/bge-m3.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Biến global cho model
_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model khi server start."""
    global _model
    from sentence_transformers import SentenceTransformer
    from config import EMBEDDING_MODEL

    print(f"🔄 Đang tải mô hình embedding: {EMBEDDING_MODEL}...")
    _model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✅ Mô hình embedding sẵn sàng ({_model.get_sentence_embedding_dimension()}D)")
    yield
    print("🛑 Embedding server đang tắt...")


app = FastAPI(title="Legal RAG Embedding Service", lifespan=lifespan)


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    vector: list[float]
    dimension: int


@app.post("/embed", response_model=EmbedResponse)
async def embed_text(req: EmbedRequest):
    """Embed một đoạn văn bản thành vector."""
    vector = _model.encode(req.text, normalize_embeddings=True).tolist()
    return EmbedResponse(vector=vector, dimension=len(vector))


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
