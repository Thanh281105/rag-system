"""
Router Agent — Embedding-based intent classification.

KHÔNG dùng LLM để classify. Thay vào đó:
1. Precompute embedding vectors cho TECHNICAL và CASUAL keyword clusters
2. Embed câu hỏi → cosine similarity với 2 cluster
3. Trả về intent dựa trên cluster gần nhất

Latency: ~10ms (so với ~500ms nếu gọi Groq).
"""
import numpy as np
from rich.console import Console

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.model_registry import get_embed_model

from utils.console import console

# Singleton
_technical_centroid = None
_casual_centroid = None

# ─── Keyword Clusters ────────────────────────────────────

TECHNICAL_KEYWORDS = [
    # AI/ML core
    "artificial intelligence", "machine learning", "deep learning",
    "neural network", "transformer", "attention mechanism",
    "language model", "large language model", "LLM",
    # Methods
    "fine-tuning", "LoRA", "QLoRA", "RLHF", "DPO", "PPO",
    "RAG", "retrieval augmented generation", "knowledge graph",
    "embedding", "vector search", "semantic search",
    "training", "inference", "optimization", "gradient descent",
    # Architecture
    "encoder", "decoder", "BERT", "GPT", "T5", "LLaMA",
    "diffusion model", "GAN", "VAE", "autoencoder",
    "convolutional", "recurrent", "RNN", "LSTM", "GRU",
    # NLP
    "tokenization", "word embedding", "sentence embedding",
    "named entity recognition", "NER", "text classification",
    "machine translation", "summarization", "question answering",
    # Computer Vision
    "image classification", "object detection", "segmentation",
    "YOLO", "ResNet", "ViT", "vision transformer",
    # Metrics
    "accuracy", "precision", "recall", "F1 score",
    "BLEU", "ROUGE", "perplexity", "loss function",
    # Vietnamese technical
    "thuật toán", "mô hình", "huấn luyện", "dữ liệu",
    "mạng nơ-ron", "học sâu", "học máy", "trí tuệ nhân tạo",
    "xử lý ngôn ngữ tự nhiên", "thị giác máy tính",
    "paper", "nghiên cứu", "bài báo", "phương pháp",
    "kiến trúc", "hiệu suất", "đánh giá", "so sánh",
]

CASUAL_KEYWORDS = [
    # Greetings
    "hello", "hi", "hey", "xin chào", "chào bạn", "chào",
    # Thanks
    "thank you", "thanks", "cảm ơn", "cám ơn",
    # Farewell
    "goodbye", "bye", "tạm biệt", "bái bai",
    # Identity
    "who are you", "bạn là ai", "tên gì", "your name",
    # Capabilities
    "what can you do", "bạn làm được gì", "bạn có thể làm được gì",
    "bạn giúp gì được", "bạn biết gì", "help me", "giúp tôi",
    "bạn có thể giúp", "có thể làm gì",
    # Smalltalk
    "how are you", "bạn khỏe không", "thời tiết",
    "weather", "joke", "funny", "haha",
    # Off-topic
    "nấu ăn", "cooking", "recipe", "music", "game",
    "phim", "movie", "thể thao", "sport", "football",
]




def _ensure_centroids():
    """Precompute cluster centroids lần đầu."""
    global _technical_centroid, _casual_centroid

    if _technical_centroid is not None:
        return

    model = get_embed_model()

    console.print("[dim]  Router: computing cluster centroids...[/]")

    # Embed all keywords
    tech_embeddings = model.encode(
        TECHNICAL_KEYWORDS, normalize_embeddings=True, batch_size=64
    )
    casual_embeddings = model.encode(
        CASUAL_KEYWORDS, normalize_embeddings=True, batch_size=64
    )

    # Compute centroids (mean of all keyword embeddings)
    _technical_centroid = np.mean(tech_embeddings, axis=0)
    _technical_centroid /= np.linalg.norm(_technical_centroid)

    _casual_centroid = np.mean(casual_embeddings, axis=0)
    _casual_centroid /= np.linalg.norm(_casual_centroid)

    console.print("[dim]  Router: centroids ready ✓[/]")


# ─── Fast-path keyword lists ────────────────────────────
# Exact/substring matches for instant classification (no GPU needed)
CASUAL_FAST_PATTERNS = [
    # Greetings (chỉ match khi là CÂU NGẮN, không dùng substring lỏng)
    "xin chào", "chào bạn", "hello", "hey",
    # Identity / capabilities
    "bạn là ai", "bạn là gì", "tên gì", "who are you",
    "bạn làm được gì", "bạn có thể làm", "bạn giúp gì",
    "bạn biết gì", "có thể làm gì", "what can you do",
    "help me", "giúp tôi",
    # Thanks / farewell
    "cảm ơn", "cám ơn", "thank", "tạm biệt", "bye",
    # Smalltalk
    "bạn khỏe", "how are you", "thời tiết",
]

# Patterns chỉ match nếu NGUYÊN CÂU rất ngắn (< 15 ký tự)
CASUAL_SHORT_ONLY = ["chào", "hi", "hey", "ok", "ừ", "vâng"]

TECHNICAL_FAST_PATTERNS = [
    # Paper / research
    "paper", "bài báo", "nghiên cứu", "arxiv",
    # Architecture / model
    "transformer", "attention", "BERT", "GPT", "LLaMA", "LoRA",
    "neural network", "mạng nơ-ron", "mô hình",
    # Methods
    "RAG", "retrieval", "fine-tuning", "training", "huấn luyện",
    "embedding", "vector", "reranking",
    # Metrics
    "accuracy", "F1", "BLEU", "benchmark",
    # Vietnamese technical (common questions)
    "thuật toán", "học sâu", "học máy", "trí tuệ nhân tạo",
    "kiến trúc", "hiệu suất", "so sánh", "đánh giá",
    "phương pháp", "dữ liệu", "dataset",
    # Broad question signals — bias toward technical
    "là gì", "hoạt động", "cách", "tại sao", "như thế nào",
    "giải thích", "khác nhau", "ưu điểm", "nhược điểm",
]


def _fast_classify(question: str) -> str | None:
    """
    Fast keyword check — O(n) string matching, ~0ms.
    Returns 'CASUAL', 'TECHNICAL', or None (ambiguous → fall through to embedding).

    PRIORITY: TECHNICAL first (bias an toàn cho ArXiv RAG system).
    """
    q_lower = question.lower().strip()

    # Stage A: Check TECHNICAL first (ưu tiên!)
    for pattern in TECHNICAL_FAST_PATTERNS:
        if pattern.lower() in q_lower:
            return "TECHNICAL"

    # Stage B: Casual short-only (chỉ match khi câu rất ngắn)
    if len(q_lower) < 15:
        for pattern in CASUAL_SHORT_ONLY:
            if q_lower.startswith(pattern) or q_lower == pattern:
                return "CASUAL"

    # Stage C: Casual substring (chỉ cho các pattern dài, ít false positive)
    for pattern in CASUAL_FAST_PATTERNS:
        if pattern in q_lower:
            return "CASUAL"

    return None  # Ambiguous → use embedding


def classify(question: str) -> str:
    """
    Phân loại intent câu hỏi: TECHNICAL hoặc CASUAL.

    2-stage classification:
    1. Fast keyword matching (~0ms) — handles obvious cases
    2. Embedding-based cosine similarity (~3-4s) — only for ambiguous cases

    BIAS: Nghiêng về TECHNICAL vì đây là ArXiv RAG system.

    Args:
        question: Câu hỏi từ user (tiếng Việt hoặc Anh)

    Returns:
        "TECHNICAL" hoặc "CASUAL"
    """
    # Stage 1: Fast keyword check
    fast_result = _fast_classify(question)
    if fast_result is not None:
        console.print(
            f"[dim]  Router: FAST → {fast_result}[/]"
        )
        return fast_result

    # Stage 2: Embedding-based (only for ambiguous questions)
    _ensure_centroids()
    model = get_embed_model()

    # Embed question
    q_embedding = model.encode(question, normalize_embeddings=True)

    # Cosine similarity
    tech_sim = float(np.dot(q_embedding, _technical_centroid))
    casual_sim = float(np.dot(q_embedding, _casual_centroid))

    # BIAS toward TECHNICAL:
    # Chỉ CASUAL khi casual_sim > tech_sim + 0.05 (margin rõ ràng)
    # Mọi trường hợp khác → TECHNICAL (an toàn hơn, vì sẽ có RAG)
    CASUAL_MARGIN = 0.05
    intent = "CASUAL" if casual_sim > tech_sim + CASUAL_MARGIN else "TECHNICAL"

    console.print(
        f"[dim]  Router: tech={tech_sim:.3f} casual={casual_sim:.3f} "
        f"(margin={CASUAL_MARGIN}) → {intent}[/]"
    )

    return intent

