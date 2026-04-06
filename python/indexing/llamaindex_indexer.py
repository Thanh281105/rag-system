"""
LlamaIndex Indexer — Embedding-only indexing vào Qdrant.

KHÔNG cần LLM. Chỉ dùng BGE-M3 embedding model để:
1. Tạo dense vector (semantic)
2. Tạo sparse vector (BM25-like)
3. Upsert vào Qdrant collection

Tốc độ: ~30 papers trong 2-3 phút (thay vì 7-20 giờ với LightRAG).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console

from config import QDRANT_COLLECTION
from retrieval.qdrant_client import QdrantWrapper
from agents.model_registry import get_embed_model

console = Console()

# Singleton instances
_qdrant: QdrantWrapper | None = None




def _get_qdrant() -> QdrantWrapper:
    """Lazy load Qdrant connection."""
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantWrapper()
        _qdrant.create_collection(recreate=False)
    return _qdrant


def index_chunk(chunk_event: dict) -> bool:
    """
    Index một chunk vào Qdrant (embedding-only, KHÔNG cần LLM).

    Args:
        chunk_event: Dict chứa text, doc_title, authors, year, arxiv_id, etc.

    Returns:
        True nếu index thành công
    """
    import numpy as np

    text = chunk_event.get("text", "")
    title = chunk_event.get("doc_title", "")
    authors = chunk_event.get("authors", "")
    year = chunk_event.get("year", "")
    arxiv_id = chunk_event.get("arxiv_id", "")

    if not text or len(text) < 50:
        console.print("[yellow]⚠️ Chunk too short, skipping[/]")
        return True

    # Inject paper context vào text cho embedding tốt hơn
    contextualized_text = (
        f"[Paper: {title} | Authors: {authors} | Year: {year} | arXiv: {arxiv_id}]\n\n"
        f"{text}"
    )

    try:
        # 1. Embed chunk (dense vector)
        model = get_embed_model()
        embedding = model.encode(
            contextualized_text,
            normalize_embeddings=True,
        )

        # 2. Prepare node for Qdrant upsert
        node = {
            "text": text,
            "node_id": chunk_event.get("chunk_id", 0),
            "level": 0,
            "doc_title": title,
            "doc_id": chunk_event.get("paper_id", 0),
            "metadata": {
                "authors": authors,
                "year": int(year) if year else 0,
                "arxiv_id": arxiv_id,
            },
        }

        # 3. Upsert vào Qdrant (dense + sparse)
        qdrant = _get_qdrant()
        qdrant.upsert_nodes(
            nodes=[node],
            embeddings=np.array([embedding]),
            batch_size=1,
        )

        return True

    except Exception as e:
        console.print(f"[red]❌ Indexing error: {e}[/]")
        return False


def index_batch(chunks: list[dict]) -> int:
    """
    Index một batch chunks vào Qdrant.

    Returns:
        Số chunks đã index thành công.
    """
    import numpy as np

    if not chunks:
        return 0

    model = get_embed_model()
    qdrant = _get_qdrant()

    # Prepare texts for batch embedding
    texts = []
    valid_chunks = []

    for chunk in chunks:
        text = chunk.get("text", "")
        if not text or len(text) < 50:
            continue

        title = chunk.get("doc_title", "")
        authors = chunk.get("authors", "")
        year = chunk.get("year", "")
        arxiv_id = chunk.get("arxiv_id", "")

        contextualized = (
            f"[Paper: {title} | Authors: {authors} | Year: {year} | arXiv: {arxiv_id}]\n\n"
            f"{text}"
        )
        texts.append(contextualized)
        valid_chunks.append(chunk)

    if not texts:
        return 0

    # Batch embed
    console.print(f"[cyan]🔄 Embedding {len(texts)} chunks...[/]")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 10,
        batch_size=32,
    )

    # Prepare nodes
    nodes = []
    for chunk in valid_chunks:
        nodes.append({
            "text": chunk.get("text", ""),
            "node_id": chunk.get("chunk_id", 0),
            "level": 0,
            "doc_title": chunk.get("doc_title", ""),
            "doc_id": chunk.get("paper_id", 0),
            "metadata": {
                "authors": chunk.get("authors", ""),
                "year": int(chunk.get("year", 0) or 0),
                "arxiv_id": chunk.get("arxiv_id", ""),
            },
        })

    # Batch upsert
    qdrant.upsert_nodes(
        nodes=nodes,
        embeddings=np.array(embeddings),
        batch_size=100,
    )

    console.print(f"[green]✅ Indexed {len(nodes)} chunks successfully[/]")
    return len(nodes)
