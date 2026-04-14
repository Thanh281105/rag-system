"""
Rewriter Agent — Viết lại query khi evidence không đủ chất lượng.

Sử dụng Groq model FAST (LLaMA-3.1-8B) cho tốc độ.
Chiến lược rewrite:
  - Thêm từ khóa kỹ thuật cụ thể hơn
  - Mở rộng viết tắt (LoRA → Low-Rank Adaptation)
  - Thay đổi góc nhìn câu hỏi
  - Thêm synonyms liên quan
"""
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.state import AgentState
from agents.llm_client import groq_complete, GROQ_MODEL_FAST

from utils.console import console


REWRITE_SYSTEM_PROMPT = """\
You are a query rewriting expert for academic paper search on ArXiv.
Given a query that returned poor search results, rewrite it to improve retrieval.

Strategies:
1. Use more specific technical terms and keywords
2. Add synonyms or closely related concepts
3. Expand abbreviations (e.g., LoRA → Low-Rank Adaptation)
4. Rephrase from a different angle to capture more relevant papers
5. Include methodology or architecture names if implied

RULES:
- Output ONLY the rewritten query in English, nothing else
- Keep it concise (1-2 sentences max)
- Preserve the core intent of the original question
"""


async def rewrite_query_node(state: AgentState) -> dict:
    """
    Node: Viết lại query EN để cải thiện retrieval.

    Sử dụng model 8B (siêu nhanh) để tinh chỉnh câu truy vấn
    khi grade_documents phát hiện evidence không đủ chất lượng.

    Returns:
        Dict cập nhật translated_query (ghi đè) và rewrite_count (+1).
    """
    t0 = time.time()
    original_query = state.get("translated_query", state["question"])
    rewrite_count = state.get("rewrite_count", 0)
    evidence = state.get("evidence", [])

    # Cung cấp context về evidence đã tìm được (nếu có)
    evidence_context = ""
    if evidence:
        titles = [doc.get("doc_title", "")[:80] for doc in evidence[:3]]
        evidence_context = (
            f"\nPrevious search returned these partially relevant papers:\n"
            + "\n".join(f"- {t}" for t in titles if t)
            + "\nRewrite to find MORE relevant papers."
        )

    prompt = (
        f"Original query (attempt #{rewrite_count + 1}): {original_query}\n"
        f"{evidence_context}\n\n"
        f"Rewrite this query to find better academic papers on ArXiv:"
    )

    rewritten = await groq_complete(
        prompt=prompt,
        system_prompt=REWRITE_SYSTEM_PROMPT,
        model=GROQ_MODEL_FAST,
        max_tokens=128,
        temperature=0.3,
    )

    rewritten = rewritten.strip().strip('"').strip("'")

    elapsed = int((time.time() - t0) * 1000)
    console.print(
        f"[yellow]  🔄 Rewrite #{rewrite_count + 1}: "
        f"'{original_query[:40]}...' → '{rewritten[:40]}...' ({elapsed}ms)[/]"
    )

    return {
        "translated_query": rewritten,
        "rewrite_count": rewrite_count + 1,
        "agent_trace": {
            **(state.get("agent_trace") or {}),
            f"rewrite_{rewrite_count + 1}_from": original_query,
            f"rewrite_{rewrite_count + 1}_to": rewritten,
            f"rewrite_{rewrite_count + 1}_ms": elapsed,
        },
    }
