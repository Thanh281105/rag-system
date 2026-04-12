"""
LangGraph Agent Orchestration — StateGraph cho RAG pipeline.

Flow:
  START → router
        ├─ CASUAL → casual_response → END
        └─ TECHNICAL → translate → retrieve → writer_stream
                                            → [conditional] reviewer → END

Entry point: run_streaming(question, history, stream_callback)
"""
import asyncio
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import Callable, Awaitable, Optional
from rich.console import Console

from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.router import classify
from agents.translator import translate_to_english
from agents.writer import generate_streaming, generate_casual
from agents.reviewer import needs_review, review_with_retry
from indexing.query_engine import retrieve_and_rerank, format_evidence

from utils.console import console


# ═══════════════════════════════════════════════════════════
# Graph Nodes
# ═══════════════════════════════════════════════════════════

def router_node(state: AgentState) -> dict:
    """Node 1: Classify intent (embedding-based, no LLM)."""
    t0 = time.time()
    question = state["question"]

    # Contextualize with history
    history = state.get("history", [])
    contextualized_q = _build_contextualized_question(question, history)

    intent = classify(contextualized_q)

    elapsed = int((time.time() - t0) * 1000)
    console.print(f"[dim]  Router: {intent} ({elapsed}ms)[/]")

    return {
        "intent": intent,
        "agent_trace": {
            **(state.get("agent_trace") or {}),
            "router_decision": intent,
            "router_ms": elapsed,
        },
    }


async def casual_node(state: AgentState) -> dict:
    """Node: Casual response (no RAG needed)."""
    question = state["question"]
    answer = await generate_casual(question)

    return {
        "answer": answer,
        "reviewer_triggered": False,
        "reviewer_result": {"is_approved": True, "issues": [], "retry_count": 0},
    }


async def translate_node(state: AgentState) -> dict:
    """Node 2: Translate VN → EN."""
    t0 = time.time()
    question = state["question"]

    translated = await translate_to_english(question)

    elapsed = int((time.time() - t0) * 1000)
    console.print(f"[dim]  Translated: '{translated[:60]}...' ({elapsed}ms)[/]")

    return {
        "translated_query": translated,
        "agent_trace": {
            **(state.get("agent_trace") or {}),
            "translated_query": translated,
            "translate_ms": elapsed,
        },
    }


def retrieve_node(state: AgentState) -> dict:
    """Node 3: Hybrid Search + Rerank."""
    t0 = time.time()
    translated_query = state.get("translated_query", state["question"])

    documents = retrieve_and_rerank(translated_query)
    evidence_text = format_evidence(documents)

    elapsed = int((time.time() - t0) * 1000)
    console.print(f"[dim]  Retrieved: {len(documents)} docs, {len(evidence_text)} chars ({elapsed}ms)[/]")

    return {
        "evidence": documents,
        "evidence_text": evidence_text,
        "agent_trace": {
            **(state.get("agent_trace") or {}),
            "retrieved_count": len(documents),
            "retrieved_context": evidence_text[:500] if evidence_text else "",
            "retrieve_ms": elapsed,
        },
    }


async def writer_node(state: AgentState) -> dict:
    """Node 4: Generate answer with streaming."""
    t0 = time.time()
    question = state["question"]
    evidence_text = state.get("evidence_text", "")
    stream_callback = state.get("stream_callback")

    answer = await generate_streaming(
        question_vn=question,
        evidence_text=evidence_text,
        stream_callback=stream_callback,
    )

    elapsed = int((time.time() - t0) * 1000)
    console.print(f"[dim]  Writer: {len(answer)} chars ({elapsed}ms)[/]")

    return {
        "answer": answer,
        "agent_trace": {
            **(state.get("agent_trace") or {}),
            "analyst_answer": answer[:500],
            "writer_ms": elapsed,
        },
    }


async def reviewer_node(state: AgentState) -> dict:
    """Node 5: Conditional fact-check."""
    t0 = time.time()
    question = state["question"]
    answer = state.get("answer", "")
    evidence_text = state.get("evidence_text", "")
    translated_query = state.get("translated_query", "")

    reviewer_triggered = needs_review(question) or needs_review(translated_query)

    if reviewer_triggered:
        console.print("[dim]  🔍 Reviewer triggered[/]")
        final_answer, reviewer_result = await review_with_retry(
            question, answer, evidence_text
        )
    else:
        console.print("[dim]  ⚡ Reviewer skipped[/]")
        final_answer = answer
        reviewer_result = {"is_approved": True, "issues": [], "retry_count": 0}

    elapsed = int((time.time() - t0) * 1000)
    console.print(f"[dim]  Reviewer: ({elapsed}ms)[/]")

    return {
        "answer": final_answer,
        "reviewer_triggered": reviewer_triggered,
        "reviewer_result": reviewer_result,
        "agent_trace": {
            **(state.get("agent_trace") or {}),
            "reviewer_triggered": reviewer_triggered,
            "reviewer_result": reviewer_result,
            "reviewer_ms": elapsed,
        },
    }


# ═══════════════════════════════════════════════════════════
# Conditional Edges
# ═══════════════════════════════════════════════════════════

def route_by_intent(state: AgentState) -> str:
    """Branch based on router classification."""
    return "casual" if state.get("intent") == "CASUAL" else "technical"


# ═══════════════════════════════════════════════════════════
# Graph Builder
# ═══════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """
    Build LangGraph StateGraph cho RAG pipeline.

    Returns:
        Compiled StateGraph
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("casual", casual_node)
    graph.add_node("translate", translate_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    # Set entry point
    graph.set_entry_point("router")

    # Conditional branching after router
    graph.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "casual": "casual",
            "technical": "translate",
        },
    )

    # Casual → END
    graph.add_edge("casual", END)

    # Technical pipeline: translate → retrieve → writer → reviewer → END
    graph.add_edge("translate", "retrieve")
    graph.add_edge("retrieve", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_edge("reviewer", END)

    return graph.compile()


# ═══════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════

# Singleton compiled graph
_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


async def run_streaming(
    question: str,
    history: list[dict] = None,
    session_id: str = "",
    stream_callback: Optional[Callable[[str], Awaitable[None]]] = None,
) -> dict:
    """
    Run full RAG pipeline with streaming.

    Args:
        question: Câu hỏi tiếng Việt
        history: Chat history
        session_id: Session ID cho tracking
        stream_callback: Async callback nhận stream tokens

    Returns:
        Final state dict chứa answer, sources, agent_trace
    """
    start_time = time.time()

    console.print(f"[cyan]💬 Processing: '{question[:60]}...'[/]")

    graph = _get_graph()

    # Initial state
    initial_state: AgentState = {
        "session_id": session_id,
        "question": question,
        "history": history or [],
        "intent": "",
        "translated_query": "",
        "evidence": [],
        "evidence_text": "",
        "answer": "",
        "reviewer_triggered": False,
        "reviewer_result": {},
        "agent_trace": {},
        "processing_time_ms": 0,
        "stream_callback": stream_callback,
    }

    # Invoke graph
    final_state = await graph.ainvoke(initial_state)

    processing_time = int((time.time() - start_time) * 1000)
    final_state["processing_time_ms"] = processing_time

    console.print(f"[green]✅ Query processed in {processing_time}ms[/]")

    return final_state


# ═══════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════

def _build_contextualized_question(question: str, history: list[dict]) -> str:
    """Xây dựng câu hỏi có ngữ cảnh từ lịch sử hội thoại."""
    if not history:
        return question

    recent = history[-6:]  # 3 cặp Q&A gần nhất

    context = "Lịch sử hội thoại gần đây:\n"
    for msg in recent:
        role = "Người hỏi" if msg.get("role") == "user" else "Trợ lý"
        content = msg.get("content", "")[:200]
        context += f"- {role}: {content}\n"

    context += f"\nCâu hỏi hiện tại: {question}"
    return context
