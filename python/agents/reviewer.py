"""
Reviewer Agent — Conditional fact-checker.

Kiểm tra câu trả lời của Writer dựa trên evidence:
- Hallucination detection
- Number/metric accuracy
- Translation correctness
- Unsupported conclusions

Chỉ trigger khi câu hỏi chứa metrics/numbers/comparisons.
"""
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from agents.llm_client import groq_complete, GROQ_MODEL_SMART

console = Console()

REVIEWER_SYSTEM_PROMPT = """\
You are a Reviewer Agent — a strict fact-checker for cross-lingual AI research Q&A.

TASK: Compare the Vietnamese ANSWER against the English SOURCE EVIDENCE and detect:
1. Hallucination — information NOT present in the evidence
2. Incorrect translation — technical terms wrongly translated
3. Wrong numbers — accuracy figures, parameter counts, dates that don't match
4. Unsupported conclusions — claims not backed by evidence
5. Shallow answer — answer only paraphrases the paper title without detail
6. Missing details — evidence contains metrics but answer omits them

RESPOND in JSON format:
{
    "is_approved": true/false,
    "issues": ["Issue 1", "Issue 2"],
    "suggestion": "Brief suggestion if there are issues"
}
"""

MAX_REVIEW_RETRIES = 2

# Heuristic keywords that trigger review
REVIEW_TRIGGER_KEYWORDS = [
    "accuracy", "f1", "precision", "recall", "bleu", "rouge",
    "bao nhiêu", "số liệu", "kết quả", "hiệu suất", "tỷ lệ",
    "parameter", "params", "flops", "latency",
    "công thức", "formula", "equation",
    "so sánh", "compare", "tốt hơn", "better", "worse",
    "chi phí", "cost", "giá",
]


def needs_review(question: str) -> bool:
    """Kiểm tra xem query có cần Reviewer không (heuristic)."""
    question_lower = question.lower()
    return any(kw in question_lower for kw in REVIEW_TRIGGER_KEYWORDS)


async def review(question: str, answer: str, evidence: str) -> dict:
    """
    Review câu trả lời dựa trên evidence.

    Returns:
        dict {is_approved: bool, issues: list, suggestion: str}
    """
    user_prompt = (
        f"QUESTION: {question}\n\n"
        f"ANSWER TO CHECK (Vietnamese):\n{answer}\n\n"
        f"SOURCE EVIDENCE (English):\n{evidence[:4000]}\n\n"
        f"Check and respond in JSON:"
    )

    response = await groq_complete(
        prompt=user_prompt,
        system_prompt=REVIEWER_SYSTEM_PROMPT,
        model=GROQ_MODEL_SMART,
        max_tokens=512,
        temperature=0.0,
    )

    return _parse_result(response)


def _parse_result(response: str) -> dict:
    """Parse JSON result từ reviewer LLM."""
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(response[start:end])
            return {
                "is_approved": result.get("is_approved", True),
                "issues": result.get("issues", []),
                "suggestion": result.get("suggestion", ""),
            }
    except (json.JSONDecodeError, KeyError):
        pass

    return {"is_approved": True, "issues": [], "suggestion": ""}


async def review_with_retry(
    question: str, answer: str, evidence: str
) -> tuple[str, dict]:
    """
    Pipeline: review → retry nếu fail.

    Returns:
        (final_answer, reviewer_result)
    """
    current_answer = answer
    retry_count = 0

    for attempt in range(MAX_REVIEW_RETRIES + 1):
        result = await review(question, current_answer, evidence)

        if result["is_approved"] or attempt >= MAX_REVIEW_RETRIES:
            result["retry_count"] = retry_count
            return current_answer, result

        console.print(
            f"[yellow]  🔄 Reviewer retry #{attempt + 1}: {result['issues']}[/]"
        )

        # Regenerate answer
        issues_str = "; ".join(result["issues"])
        retry_prompt = (
            f"Câu trả lời trước đã bị phát hiện lỗi: {issues_str}\n\n"
            f"CÂU HỎI: {question}\n\n"
            f"EVIDENCE: {evidence[:4000]}\n\n"
            f"Hãy viết lại câu trả lời bằng tiếng Việt, tránh các lỗi trên. "
            f"Giữ thuật ngữ kỹ thuật bằng tiếng Anh."
        )

        current_answer = await groq_complete(
            prompt=retry_prompt,
            system_prompt="Bạn là chuyên gia AI. Viết câu trả lời chính xác bằng tiếng Việt.",
            model=GROQ_MODEL_SMART,
            temperature=0.1,
        )
        retry_count += 1

    return current_answer, result
