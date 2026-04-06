"""
Agent State — Shared state definition cho LangGraph pipeline.
Mỗi node trong graph đọc/ghi vào state này.
"""
from typing import Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """State chung cho toàn bộ LangGraph pipeline."""

    # ─── Input ────────────────────────────────────────
    session_id: str
    question: str               # Câu hỏi VN gốc
    history: list[dict]         # Chat history [{role, content}, ...]

    # ─── Router Output ────────────────────────────────
    intent: str                 # "TECHNICAL" | "CASUAL"

    # ─── Translation Output ───────────────────────────
    translated_query: str       # Câu hỏi đã dịch sang EN

    # ─── Retrieval Output ─────────────────────────────
    evidence: list[dict]        # Retrieved & reranked chunks
    evidence_text: str          # Formatted evidence text cho LLM

    # ─── Generation Output ────────────────────────────
    answer: str                 # Câu trả lời cuối cùng (full)

    # ─── Reviewer Output ──────────────────────────────
    reviewer_triggered: bool
    reviewer_result: dict       # {is_approved, issues, retry_count}

    # ─── Metadata ─────────────────────────────────────
    agent_trace: dict           # Debug trace cho frontend
    processing_time_ms: int

    # ─── Streaming ────────────────────────────────────
    stream_callback: Any        # Callable[[str], Awaitable[None]]
