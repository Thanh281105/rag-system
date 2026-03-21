"""
Embedding module sử dụng BAAI/bge-m3 cho văn bản pháp lý tiếng Việt.
bge-m3 hỗ trợ đa ngôn ngữ (bao gồm tiếng Việt) và output cả dense + sparse vectors.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import track

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import EMBEDDING_MODEL, EMBEDDING_DIM

console = Console()

# Cache model singleton
_model = None


def get_model() -> SentenceTransformer:
    """Lazy load embedding model (singleton)."""
    global _model
    if _model is None:
        console.print(f"[cyan]🔄 Đang tải mô hình embedding: {EMBEDDING_MODEL}...[/]")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        console.print(f"[green]✅ Đã tải mô hình embedding ({EMBEDDING_DIM}D)[/]")
    return _model


def embed_texts(texts: list[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
    """
    Nhúng danh sách văn bản thành vectors.
    
    Args:
        texts: Danh sách chuỗi văn bản
        batch_size: Kích thước batch
        show_progress: Hiển thị thanh tiến trình
        
    Returns:
        np.ndarray shape (n, EMBEDDING_DIM) - Ma trận embedding
    """
    model = get_model()
    
    if show_progress:
        console.print(f"[cyan]🔢 Đang nhúng {len(texts)} đoạn văn bản...[/]")
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # L2 normalize cho cosine similarity
    )
    
    if show_progress:
        console.print(f"[green]✅ Đã tạo {embeddings.shape[0]} vectors ({embeddings.shape[1]}D)[/]")
    
    return embeddings


def embed_single(text: str) -> np.ndarray:
    """Nhúng 1 văn bản duy nhất."""
    model = get_model()
    embedding = model.encode(
        text,
        normalize_embeddings=True,
    )
    return embedding


if __name__ == "__main__":
    # Test embedding
    test_texts = [
        "Điều 1. Phạm vi điều chỉnh của Luật Doanh nghiệp",
        "Quy định về thành lập và tổ chức quản lý doanh nghiệp",
        "Hôm nay trời đẹp quá",  # Câu không liên quan để test
    ]
    
    vectors = embed_texts(test_texts, show_progress=False)
    
    # Tính cosine similarity
    sim_01 = np.dot(vectors[0], vectors[1])
    sim_02 = np.dot(vectors[0], vectors[2])
    
    console.print(f"[bold]Similarity test:[/]")
    console.print(f"  Pháp lý ↔ Pháp lý: {sim_01:.4f}")
    console.print(f"  Pháp lý ↔ Thông thường: {sim_02:.4f}")
    console.print(f"  → Embedding phân biệt tốt: {'✅' if sim_01 > sim_02 else '❌'}")
