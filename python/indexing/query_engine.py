"""
Query Engine — Hybrid Retrieval + Reranking.

Tái sử dụng retrieval/hybrid_search.py + retrieval/reranker.py.
Cung cấp interface đơn giản cho LangGraph retrieve node.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from rich.console import Console

from config import (
    ENABLE_FEEDBACK_RETRY,
    ENABLE_METADATA_RETRIEVAL,
    FEEDBACK_MAX_RETRIEVAL_RETRIES,
    METADATA_DOCS_PER_MATCH,
    TOP_K_RETRIEVAL,
    TOP_K_RERANK,
    USE_RERANKER,
)
from retrieval.feedback import (
    build_retry_query,
    is_low_confidence,
    retrieval_confidence,
    write_feedback_event,
)
from retrieval.qdrant_client import QdrantWrapper
from retrieval.hybrid_search import hybrid_search
from retrieval.metadata_search import merge_metadata_and_hybrid, metadata_search
from retrieval.reranker import rerank
from agents.model_registry import get_embed_model

from utils.console import console

# Singleton
_qdrant: QdrantWrapper | None = None




def _get_qdrant() -> QdrantWrapper:
    """Lazy load Qdrant connection."""
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantWrapper()
    return _qdrant


def retrieve(
    query_en: str,
    top_k: int = TOP_K_RETRIEVAL,
    original_query: str = "",
) -> list[dict]:
    """
    Hybrid search (Dense + Sparse + RRF) trên Qdrant.

    Args:
        query_en: Query đã dịch sang tiếng Anh
        top_k: Số kết quả trả về

    Returns:
        List[dict] — mỗi dict chứa text, doc_title, authors, year, arxiv_id, rrf_score
    """
    model = get_embed_model()
    qdrant = _get_qdrant()
    metadata_query = "\n".join(q for q in [original_query, query_en] if q)

    metadata_candidates = []
    metadata_docs = []
    if ENABLE_METADATA_RETRIEVAL:
        metadata_candidates = metadata_search(metadata_query)
        if metadata_candidates:
            arxiv_ids = [c["arxiv_id"] for c in metadata_candidates if c.get("arxiv_id")]
            metadata_docs = qdrant.scroll_by_arxiv_ids(
                arxiv_ids,
                limit_per_id=METADATA_DOCS_PER_MATCH,
            )
            console.print(
                f"[dim]  Metadata: {len(metadata_candidates)} candidates, "
                f"{len(metadata_docs)} indexed chunks[/]"
            )

    # Embed query
    query_vector = model.encode(query_en, normalize_embeddings=True)

    # Hybrid search
    results = hybrid_search(
        query_vector=np.array(query_vector),
        query_text=query_en,
        qdrant=qdrant,
        top_k=top_k,
    )

    if ENABLE_METADATA_RETRIEVAL and (metadata_candidates or metadata_docs):
        results = merge_metadata_and_hybrid(
            metadata_docs=metadata_docs,
            hybrid_docs=results,
            candidates=metadata_candidates,
            top_k=top_k,
        )

    return results


def retrieve_and_rerank(
    query_en: str,
    top_k_search: int = TOP_K_RETRIEVAL,
    top_k_rerank: int = TOP_K_RERANK,
    original_query: str = "",
) -> list[dict]:
    """
    Hybrid Search → Cross-encoder Reranking.

    Pipeline:
    1. Hybrid search (Dense + Sparse + RRF) → top_k_search candidates
    2. Cross-encoder reranking (BGE-Reranker-v2-M3) → top_k_rerank final

    Args:
        query_en: Query tiếng Anh
        top_k_search: Candidates từ hybrid search
        top_k_rerank: Kết quả cuối sau reranking

    Returns:
        Top-K reranked documents
    """
    # Step 1: Hybrid search
    candidates = retrieve(query_en, top_k=top_k_search, original_query=original_query)

    if not candidates:
        console.print("[yellow]⚠️ No results from hybrid search[/]")
        return []

    if not USE_RERANKER:
        results = candidates[:top_k_rerank]
        console.print(
            f"[dim]  Reranker disabled: using top {len(results)} RRF results[/]"
        )
        return results

    # Step 2: Rerank
    reranked = rerank(
        query=query_en,
        documents=candidates,
        top_k=top_k_rerank,
    )

    return reranked


def retrieve_with_feedback(
    query_en: str,
    original_query: str = "",
    top_k_search: int = TOP_K_RETRIEVAL,
    top_k_rerank: int = TOP_K_RERANK,
) -> tuple[list[dict], dict]:
    """
    Retrieval with one bounded deterministic retry when confidence is low.

    This is not an agent loop. It only enriches the retrieval query with the
    original user wording and metadata aliases, then keeps the better result.
    """
    attempts = []
    documents = retrieve_and_rerank(
        query_en,
        top_k_search=top_k_search,
        top_k_rerank=top_k_rerank,
        original_query=original_query,
    )
    attempts.append({
        "attempt": 0,
        "confidence": retrieval_confidence(documents),
        "doc_count": len(documents),
        "query": query_en[:300],
    })

    best_docs = documents
    retry_count = 0

    if ENABLE_FEEDBACK_RETRY and is_low_confidence(documents):
        for retry_idx in range(FEEDBACK_MAX_RETRIEVAL_RETRIES):
            retry_query = build_retry_query(query_en, original_query)
            retry_docs = retrieve_and_rerank(
                retry_query,
                top_k_search=top_k_search,
                top_k_rerank=top_k_rerank,
                original_query=original_query,
            )
            retry_count += 1
            retry_confidence = retrieval_confidence(retry_docs)
            attempts.append({
                "attempt": retry_idx + 1,
                "confidence": retry_confidence,
                "doc_count": len(retry_docs),
                "query": retry_query[:300],
            })

            if retry_confidence > retrieval_confidence(best_docs):
                best_docs = retry_docs
                break

    low_confidence = is_low_confidence(best_docs)
    trace = {
        "retrieval_confidence": retrieval_confidence(best_docs),
        "retrieval_low_confidence": low_confidence,
        "retrieval_feedback_retry_count": retry_count,
        "retrieval_attempts": attempts,
    }

    if low_confidence or retry_count:
        write_feedback_event({
            "event": "retrieval_feedback",
            "original_query": original_query,
            "translated_query": query_en,
            "low_confidence": low_confidence,
            "retry_count": retry_count,
            "attempts": attempts,
            "top_docs": [
                {
                    "arxiv_id": doc.get("arxiv_id", ""),
                    "title": doc.get("doc_title", ""),
                    "combined_score": doc.get("combined_score", 0),
                    "rrf_score": doc.get("rrf_score", 0),
                    "metadata_score": doc.get("metadata_score", 0),
                }
                for doc in best_docs[:3]
            ],
        })

    return best_docs, trace


def format_evidence(documents: list[dict]) -> str:
    """
    Format retrieved documents thành text evidence cho LLM prompt.

    Returns:
        Formatted evidence string
    """
    if not documents:
        return ""

    parts = []
    for i, doc in enumerate(documents, 1):
        title = doc.get("doc_title", "Unknown")
        authors = doc.get("authors", "")
        year = doc.get("year", "")
        text = doc.get("text", "")
        score = doc.get("rerank_score", doc.get("combined_score", doc.get("rrf_score", 0)))
        channels = ",".join(doc.get("retrieval_channels", []))

        parts.append(
            f"[Source {i}] {title}\n"
            f"Authors: {authors} | Year: {year} | Relevance: {score:.4f}"
            f"{f' | Channels: {channels}' if channels else ''}\n"
            f"{text}\n"
        )

    return "\n---\n".join(parts)
