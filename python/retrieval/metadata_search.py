"""
Metadata-aware retrieval helpers.

This layer is intentionally lightweight: it uses local paper metadata to detect
high-confidence title/arXiv/author/year signals, then boosts matching Qdrant
chunks instead of replacing hybrid search.
"""
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    METADATA_BOOST,
    METADATA_MIN_TITLE_OVERLAP,
    METADATA_TOP_K,
    RAW_DATA_DIR,
)


_ARXIV_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "bai", "bang", "bao", "by", "cua",
    "cho", "data", "de", "for", "from", "gi", "in", "la", "mot", "of",
    "on", "paper", "the", "to", "trong", "using", "va", "ve", "voi",
}


def normalize_text(text: str) -> str:
    """Lowercase, remove accents, and normalize whitespace/punctuation."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(normalize_text(text)) if t not in _STOPWORDS and len(t) > 1}


@lru_cache(maxsize=1)
def load_paper_metadata() -> list[dict]:
    path = RAW_DATA_DIR / "arxiv_papers.json"
    if not path.exists():
        return []

    try:
        papers = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    metadata = []
    for paper in papers:
        title = paper.get("title", "")
        authors = paper.get("authors", "")
        arxiv_id = paper.get("arxiv_id", "")
        year = paper.get("year", 0)
        title_norm = normalize_text(title)
        author_norm = normalize_text(authors)
        metadata.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "title_norm": title_norm,
            "title_tokens": _tokens(title),
            "authors": authors,
            "author_norm": author_norm,
            "year": year,
        })
    return metadata


def metadata_search(query_text: str, top_k: int = METADATA_TOP_K) -> list[dict]:
    """
    Return likely paper metadata matches for a query.

    Scores are deliberately coarse; they are used for boosting and exact
    payload fetches, not as final semantic relevance.
    """
    query_norm = normalize_text(query_text)
    query_tokens = _tokens(query_text)
    arxiv_ids = {m.group(0).lower() for m in _ARXIV_RE.finditer(query_text or "")}
    year_hits = set(re.findall(r"\b(?:19|20)\d{2}\b", query_text or ""))

    matches = []
    for paper in load_paper_metadata():
        score = 0.0
        reasons = []
        arxiv_id = paper["arxiv_id"].lower()

        if arxiv_id and arxiv_id in arxiv_ids:
            score += 1.0
            reasons.append("arxiv_id")

        title_norm = paper["title_norm"]
        if title_norm and len(title_norm) >= 12 and title_norm in query_norm:
            score += 0.95
            reasons.append("title_exact")
        else:
            title_tokens = paper["title_tokens"]
            if title_tokens:
                overlap = len(title_tokens & query_tokens) / max(len(title_tokens), 1)
                if overlap >= METADATA_MIN_TITLE_OVERLAP and len(title_tokens & query_tokens) >= 3:
                    score += 0.35 + (0.4 * overlap)
                    reasons.append(f"title_overlap:{overlap:.2f}")

        if paper["author_norm"]:
            for author in paper["author_norm"].split(","):
                author = author.strip()
                if len(author) >= 8 and author in query_norm:
                    score += 0.6
                    reasons.append("author")
                    break

        if paper["year"] and str(paper["year"]) in year_hits and score > 0:
            score += 0.1
            reasons.append("year")

        if score > 0:
            matches.append({
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "authors": paper["authors"],
                "year": paper["year"],
                "metadata_score": min(score, 1.0),
                "metadata_match_reasons": reasons,
            })

    matches.sort(key=lambda item: item["metadata_score"], reverse=True)
    return matches[:top_k]


def apply_metadata_boost(doc: dict, candidates: list[dict]) -> dict:
    boosted = doc.copy()
    doc_arxiv_id = (boosted.get("arxiv_id") or "").lower()
    match = next(
        (c for c in candidates if (c.get("arxiv_id") or "").lower() == doc_arxiv_id),
        None,
    )
    if not match:
        boosted["combined_score"] = boosted.get("combined_score", boosted.get("rrf_score", 0))
        return boosted

    metadata_score = float(match.get("metadata_score", 0))
    boosted["metadata_score"] = metadata_score
    boosted["metadata_match_reasons"] = match.get("metadata_match_reasons", [])
    boosted["metadata_matched_title"] = match.get("title", "")
    boosted["combined_score"] = (
        float(boosted.get("rrf_score", boosted.get("combined_score", 0)) or 0)
        + (metadata_score * METADATA_BOOST)
    )
    channels = set(boosted.get("retrieval_channels", []))
    channels.add("metadata")
    boosted["retrieval_channels"] = sorted(channels)
    return boosted


def merge_metadata_and_hybrid(
    metadata_docs: list[dict],
    hybrid_docs: list[dict],
    candidates: list[dict],
    top_k: int,
) -> list[dict]:
    merged = {}

    for doc in hybrid_docs:
        boosted = apply_metadata_boost(doc, candidates)
        channels = set(boosted.get("retrieval_channels", []))
        channels.add("hybrid")
        boosted["retrieval_channels"] = sorted(channels)
        merged[boosted["id"]] = boosted

    for doc in metadata_docs:
        boosted = apply_metadata_boost(doc, candidates)
        boosted.setdefault("rrf_score", 0.0)
        boosted.setdefault("combined_score", boosted.get("metadata_score", 0.0))
        channels = set(boosted.get("retrieval_channels", []))
        channels.add("metadata")
        boosted["retrieval_channels"] = sorted(channels)
        existing = merged.get(boosted["id"])
        if existing:
            existing_channels = set(existing.get("retrieval_channels", []))
            existing_channels.update(channels)
            existing["retrieval_channels"] = sorted(existing_channels)
            existing["combined_score"] = max(
                float(existing.get("combined_score", 0) or 0),
                float(boosted.get("combined_score", 0) or 0),
            )
            existing.setdefault("metadata_score", boosted.get("metadata_score", 0))
            existing.setdefault("metadata_match_reasons", boosted.get("metadata_match_reasons", []))
        else:
            merged[boosted["id"]] = boosted

    results = list(merged.values())
    results.sort(
        key=lambda doc: (
            float(doc.get("combined_score", 0) or 0),
            float(doc.get("metadata_score", 0) or 0),
            float(doc.get("rrf_score", 0) or 0),
        ),
        reverse=True,
    )
    return results[:top_k]
