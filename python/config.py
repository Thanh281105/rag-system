"""
Cấu hình chung cho toàn bộ hệ thống Cross-lingual ArXiv RAG.
Đọc biến môi trường từ file .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Paths ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Tạo thư mục nếu chưa tồn tại
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ─── Load .env ───────────────────────────────────────────
load_dotenv(ENV_FILE)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Default profile for this project target: a 4GB VRAM local GPU.
RAG_LOW_VRAM_MODE = _env_bool("RAG_LOW_VRAM_MODE", True)

# ─── HuggingFace Login ───────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN)
    except Exception:
        pass

# ─── API Keys ────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Hỗ trợ nhiều key cách nhau bằng dấu phẩy
GROQ_API_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", GROQ_API_KEY).split(",") if k.strip()]
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# ─── Model Configuration ────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    os.getenv(
        "GROQ_MODEL",
        GROQ_FAST_MODEL if RAG_LOW_VRAM_MODE else "llama-3.3-70b-versatile",
    ),
)

# Local model runtime. Keep BGE-M3 because the existing Qdrant vectors are 1024D.
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cpu" if RAG_LOW_VRAM_MODE else "auto")
EMBEDDING_FP16 = _env_bool("EMBEDDING_FP16", RAG_LOW_VRAM_MODE)
RERANKER_FP16 = _env_bool("RERANKER_FP16", False)
EMBEDDING_MAX_SEQ_LENGTH = int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "512"))
RERANKER_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "384" if RAG_LOW_VRAM_MODE else "512"))
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "4" if RAG_LOW_VRAM_MODE else "16"))
USE_RERANKER = _env_bool("USE_RERANKER", not RAG_LOW_VRAM_MODE)
ROUTER_EMBEDDING_FALLBACK = _env_bool("ROUTER_EMBEDDING_FALLBACK", not RAG_LOW_VRAM_MODE)

# ─── ArXiv Configuration ────────────────────────────────
ARXIV_TOPIC = os.getenv("ARXIV_TOPIC", "cs.AI")
ARXIV_MAX_PAPERS = int(os.getenv("ARXIV_MAX_PAPERS", "50"))

# ─── Kafka Configuration ────────────────────────────────
KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "localhost:9092")

# ─── LangGraph Configuration ────────────────────────────
LANGGRAPH_MAX_RETRIES = int(os.getenv("LANGGRAPH_MAX_RETRIES", "2"))
STREAMING_CHUNK_SIZE = int(os.getenv("STREAMING_CHUNK_SIZE", "24" if RAG_LOW_VRAM_MODE else "3"))

# ─── Qdrant Configuration ───────────────────────────────
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "arxiv_raptor")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))  # bge-m3 output dimension

# ─── Groq Configuration ─────────────────────────────────
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "1024" if RAG_LOW_VRAM_MODE else "4096"))
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.1"))  # Low temperature cho câu trả lời chính xác
GROQ_RATE_LIMIT_DELAY_SECONDS = float(os.getenv("GROQ_RATE_LIMIT_DELAY_SECONDS", "2"))
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "1024" if RAG_LOW_VRAM_MODE else "2048"))
EVIDENCE_MAX_CHARS = int(os.getenv("EVIDENCE_MAX_CHARS", "3500" if RAG_LOW_VRAM_MODE else "6000"))

# ─── Retrieval Configuration ────────────────────────────
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "6" if RAG_LOW_VRAM_MODE else "10"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "3" if RAG_LOW_VRAM_MODE else "5"))
ENABLE_METADATA_RETRIEVAL = _env_bool("ENABLE_METADATA_RETRIEVAL", True)
METADATA_TOP_K = int(os.getenv("METADATA_TOP_K", "5"))
METADATA_DOCS_PER_MATCH = int(os.getenv("METADATA_DOCS_PER_MATCH", "3"))
METADATA_MIN_TITLE_OVERLAP = float(os.getenv("METADATA_MIN_TITLE_OVERLAP", "0.55"))
METADATA_BOOST = float(os.getenv("METADATA_BOOST", "1.0"))
ENABLE_FEEDBACK_RETRY = _env_bool("ENABLE_FEEDBACK_RETRY", True)
FEEDBACK_MAX_RETRIEVAL_RETRIES = int(os.getenv("FEEDBACK_MAX_RETRIEVAL_RETRIES", "1"))
LOW_CONFIDENCE_MIN_DOCS = int(os.getenv("LOW_CONFIDENCE_MIN_DOCS", "1"))
LOW_CONFIDENCE_MIN_SCORE = float(os.getenv("LOW_CONFIDENCE_MIN_SCORE", "0.015"))
FEEDBACK_LOG_PATH = os.getenv(
    "FEEDBACK_LOG_PATH",
    str(DATA_DIR / "feedback" / "retrieval_feedback.jsonl"),
)

# ─── RRF (Reciprocal Rank Fusion) Configuration ─────────
RRF_DENSE_WEIGHT = float(os.getenv("RRF_DENSE_WEIGHT", "0.6"))
RRF_SPARSE_WEIGHT = float(os.getenv("RRF_SPARSE_WEIGHT", "0.4"))
RRF_K = int(os.getenv("RRF_K", "60"))

# ─── Upstash Redis (Serverless Cache) ───────────────────
UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL", "")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", str(7 * 24 * 3600)))  # 7 ngày

# ─── Langfuse (Observability) ───────────────────────────
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# ─── Rewrite / Self-Reflective RAG ──────────────────────
MAX_REWRITE_RETRIES = int(os.getenv("MAX_REWRITE_RETRIES", "0" if RAG_LOW_VRAM_MODE else "2"))
MIN_GOOD_DOCS = int(os.getenv("MIN_GOOD_DOCS", "1" if RAG_LOW_VRAM_MODE else "3"))
GRADE_SCORE_THRESHOLD = float(os.getenv("GRADE_SCORE_THRESHOLD", "0.25"))
RRF_GRADE_SCORE_THRESHOLD = float(os.getenv("RRF_GRADE_SCORE_THRESHOLD", "0.0"))
ENABLE_REVIEWER = _env_bool("ENABLE_REVIEWER", not RAG_LOW_VRAM_MODE)
