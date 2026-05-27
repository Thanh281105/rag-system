"""
Bounded retrieval feedback utilities.

Online feedback is intentionally conservative: at most one deterministic retry
when retrieval confidence is low. Offline feedback is appended to JSONL for
later eval, synonym, or metadata-alias work.
"""
import json
import time
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    FEEDBACK_LOG_PATH,
    LOW_CONFIDENCE_MIN_DOCS,
    LOW_CONFIDENCE_MIN_SCORE,
)
from retrieval.metadata_search import metadata_search


def retrieval_confidence(documents: list[dict]) -> float:
    if not documents:
        return 0.0

    top = documents[0]
    return max(
        float(top.get("combined_score", 0) or 0),
        float(top.get("metadata_score", 0) or 0),
        float(top.get("rerank_score", 0) or 0),
        float(top.get("rrf_score", 0) or 0),
    )


def is_low_confidence(documents: list[dict]) -> bool:
    if len(documents) < LOW_CONFIDENCE_MIN_DOCS:
        return True
    return retrieval_confidence(documents) < LOW_CONFIDENCE_MIN_SCORE


def build_retry_query(query_en: str, original_query: str = "") -> str:
    parts = [query_en]
    if original_query and original_query.strip() != query_en.strip():
        parts.append(f"Original user question: {original_query}")

    metadata_candidates = metadata_search("\n".join(parts), top_k=3)
    if metadata_candidates:
        aliases = []
        for candidate in metadata_candidates:
            aliases.append(
                f"{candidate.get('title', '')} {candidate.get('arxiv_id', '')}".strip()
            )
        parts.append("Metadata aliases: " + " | ".join(a for a in aliases if a))

    return "\n".join(parts)


def write_feedback_event(event: dict) -> None:
    path = Path(FEEDBACK_LOG_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.time(),
                **event,
            }, ensure_ascii=False) + "\n")
    except Exception:
        # Feedback logs must never break the serving path.
        pass
