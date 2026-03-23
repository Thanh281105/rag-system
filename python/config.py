"""
Cấu hình chung cho toàn bộ hệ thống RAG pháp lý.
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
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"

# Tạo thư mục nếu chưa tồn tại
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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

# ─── RAPTOR Configuration ───────────────────────────────
RAPTOR_MAX_LEVELS = int(os.getenv("RAPTOR_MAX_LEVELS", "3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# ─── Qdrant Configuration ───────────────────────────────
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "legal_raptor")
EMBEDDING_DIM = 1024  # bge-m3 output dimension

# ─── Groq Configuration ─────────────────────────────────
GROQ_MAX_TOKENS = 4096
GROQ_TEMPERATURE = 0.1  # Low temperature cho câu trả lời pháp lý chính xác

# ─── Retrieval Configuration ────────────────────────────
TOP_K_RETRIEVAL = 20   # Số lượng kết quả hybrid search
TOP_K_RERANK = 5       # Số lượng kết quả sau reranking

# ─── RRF (Reciprocal Rank Fusion) Configuration ─────────
# Trọng số cho Hybrid Search: dense (ngữ nghĩa) vs sparse (keyword/BM25)
# Domain pháp lý VN có mật độ keyword cao → nên tăng sparse weight
RRF_DENSE_WEIGHT = float(os.getenv("RRF_DENSE_WEIGHT", "0.5"))
RRF_SPARSE_WEIGHT = float(os.getenv("RRF_SPARSE_WEIGHT", "0.5"))
RRF_K = int(os.getenv("RRF_K", "60"))  # Hằng số RRF (cao hơn → ít phân biệt rank)
