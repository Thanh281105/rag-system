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

# ─── HuggingFace Login ───────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN)
    except ImportError:
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
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# ─── ArXiv Configuration ────────────────────────────────
ARXIV_TOPIC = os.getenv("ARXIV_TOPIC", "cs.AI")
ARXIV_MAX_PAPERS = int(os.getenv("ARXIV_MAX_PAPERS", "50"))

# ─── Kafka Configuration ────────────────────────────────
KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "localhost:9092")

# ─── LangGraph Configuration ────────────────────────────
LANGGRAPH_MAX_RETRIES = 2
STREAMING_CHUNK_SIZE = 3  # Số tokens gom lại trước khi gửi qua Kafka

# ─── Qdrant Configuration ───────────────────────────────
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "arxiv_raptor")
EMBEDDING_DIM = 1024  # bge-m3 output dimension

# ─── Groq Configuration ─────────────────────────────────
GROQ_MAX_TOKENS = 4096
GROQ_TEMPERATURE = 0.1  # Low temperature cho câu trả lời chính xác

# ─── Retrieval Configuration ────────────────────────────
TOP_K_RETRIEVAL = 10   # Số lượng kết quả hybrid search (giảm từ 20 → 10 cho P1000)
TOP_K_RERANK = 5       # Số lượng kết quả sau reranking

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
MAX_REWRITE_RETRIES = 2           # Tối đa 2 lần rewrite (3 lần retrieve)
MIN_GOOD_DOCS = 3                 # Cần tối thiểu 3 docs có score tốt
GRADE_SCORE_THRESHOLD = 0.25      # Ngưỡng rerank_score coi là "tốt"
