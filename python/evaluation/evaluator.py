"""
Evaluator module: Đánh giá hệ thống Cross-lingual ArXiv RAG.
Metrics:
- Context Precision: ngữ cảnh truy xuất có liên quan không
- Context Recall: ngữ cảnh có đủ để suy ra đáp án không
- Faithfulness: câu trả lời có trung thực với ngữ cảnh không
- Answer Relevancy: câu trả lời có phù hợp với câu hỏi không
- Answer Correctness: câu trả lời có khớp với ground truth không
- Hallucination Rate: tỷ lệ thông tin bịa đặt
- Translation Faithfulness: dịch cross-lingual có chính xác không
"""
import json
import time
from groq import Groq
from rich.console import Console
from rich.table import Table
from rich.progress import track

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import GROQ_API_KEY, LLM_MODEL, SYNTHETIC_DATA_DIR

console = Console()

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def _llm_score(prompt: str) -> float:
    """Helper: gọi LLM để chấm điểm, trả về float 0.0-1.0."""
    client = get_client()
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        score_str = response.choices[0].message.content.strip()
        import re
        match = re.search(r'(\d+\.?\d*)', score_str)
        if match:
            score = float(match.group(1))
            return min(max(score, 0.0), 1.0)
        return 0.0
    except (ValueError, Exception):
        return 0.0


def evaluate_faithfulness(answer: str, contexts: list[str]) -> float:
    """Câu trả lời có đúng với ngữ cảnh hay không."""
    context_str = "\n\n".join(contexts)
    prompt = f"""Evaluate the faithfulness of the answer compared to the context.
The answer is in Vietnamese and the context is in English (cross-lingual system).

Context: {context_str[:3000]}

Answer: {answer}

Score from 0.0 to 1.0:
- 1.0: All information in the answer is supported by the context
- 0.5: Some information is correct, some is hallucinated
- 0.0: Completely fabricated

Reply with ONLY a decimal number (e.g., 0.85):"""
    return _llm_score(prompt)


def evaluate_context_precision(question: str, contexts: list[str], ground_truth: str) -> float:
    """Ngữ cảnh truy xuất có liên quan không."""
    context_str = "\n\n".join(f"[Context {i+1}]: {c}" for i, c in enumerate(contexts))
    prompt = f"""Evaluate precision of retrieved contexts for the question.

Question: {question}
Ground truth: {ground_truth}

Retrieved contexts:
{context_str[:3000]}

Score from 0.0 to 1.0:
- 1.0: All contexts are relevant
- 0.5: Half are relevant
- 0.0: None are relevant

Reply with ONLY a decimal number:"""
    return _llm_score(prompt)


def evaluate_context_recall(question: str, contexts: list[str], ground_truth: str) -> float:
    """Ngữ cảnh có đủ thông tin để suy ra đáp án không."""
    context_str = "\n\n".join(contexts)
    prompt = f"""Evaluate context recall — can the ground truth be derived from the contexts?

Question: {question}
Ground truth: {ground_truth[:1500]}

Retrieved contexts:
{context_str[:3000]}

Score from 0.0 to 1.0:
- 1.0: All information in ground truth can be derived from contexts
- 0.5: Only partial information is available
- 0.0: No relevant information in contexts

Reply with ONLY a decimal number:"""
    return _llm_score(prompt)


def evaluate_answer_relevancy(question: str, answer: str) -> float:
    """Câu trả lời có phù hợp với câu hỏi không."""
    prompt = f"""Evaluate how well the answer addresses the question.
Note: Question is in Vietnamese and answer should also be in Vietnamese.

Question: {question}
Answer: {answer}

Score from 0.0 to 1.0:
- 1.0: Complete and accurate answer
- 0.5: Partial answer
- 0.0: Irrelevant

Reply with ONLY a decimal number:"""
    return _llm_score(prompt)


def evaluate_answer_correctness(question: str, answer: str, ground_truth: str) -> float:
    """Câu trả lời có khớp với ground truth không."""
    prompt = f"""Compare the answer to the ground truth and evaluate correctness.

Question: {question}
Ground truth: {ground_truth[:1500]}
Answer to evaluate: {answer[:1500]}

Score from 0.0 to 1.0:
- 1.0: Answer and ground truth convey the same information
- 0.5: Partially correct but missing or wrong in some parts
- 0.0: Completely wrong compared to ground truth

Reply with ONLY a decimal number:"""
    return _llm_score(prompt)


def evaluate_hallucination_rate(answer: str, contexts: list[str]) -> float:
    """Tỷ lệ hallucination (0 = tốt, 1 = xấu)."""
    context_str = "\n\n".join(contexts)
    prompt = f"""Analyze hallucination rate in the answer for a cross-lingual AI research Q&A system.
The answer is in Vietnamese, the source contexts are in English.

Source contexts (English):
{context_str[:3000]}

Answer to check (Vietnamese):
{answer[:1500]}

Special checks for technical content:
- Are accuracy numbers, F1 scores, parameter counts correct?
- Are model/method names accurate?
- Are technical claims supported by the source?

Rate HALLUCINATION proportion from 0.0 to 1.0:
- 0.0: No hallucination (GOOD)
- 0.3: Minor details fabricated
- 0.5: About half is fabricated
- 1.0: Completely fabricated (BAD)

Reply with ONLY a decimal number:"""
    return _llm_score(prompt)


def evaluate_translation_faithfulness(answer_vn: str, contexts_en: list[str]) -> float:
    """Đánh giá chất lượng dịch cross-lingual: thuật ngữ kỹ thuật có được giữ nguyên không."""
    context_str = "\n\n".join(contexts_en)
    prompt = f"""Evaluate translation quality in a cross-lingual Q&A system.
The source is English technical papers and the answer is in Vietnamese.

Check:
1. Are technical terms preserved in English? (Transformer, attention, LoRA, etc.)
2. Is the meaning accurately conveyed?
3. Are citations and references correctly translated?

Source (English): {context_str[:2000]}
Answer (Vietnamese): {answer_vn[:1500]}

Score from 0.0 to 1.0:
- 1.0: Perfect cross-lingual transfer, technical terms preserved
- 0.5: Some terms incorrectly translated or meaning distorted
- 0.0: Completely wrong translation

Reply with ONLY a decimal number:"""
    return _llm_score(prompt)


def run_evaluation(
    qa_pairs: list[dict] = None,
    rag_pipeline: callable = None,
    max_samples: int = 50,
) -> dict:
    """Chạy đánh giá toàn diện trên synthetic data."""
    if qa_pairs is None:
        qa_path = SYNTHETIC_DATA_DIR / "synthetic_qa.json"
        if not qa_path.exists():
            console.print("[red]❌ No synthetic data. Run synthetic_generator first.[/]")
            return {}
        with open(qa_path, "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)

    samples = qa_pairs[:max_samples]
    console.print(f"[bold cyan]📊 Evaluating {len(samples)} samples (7 metrics)...[/]")

    results = {
        "faithfulness": [],
        "context_precision": [],
        "context_recall": [],
        "answer_relevancy": [],
        "answer_correctness": [],
        "hallucination_rate": [],
        "translation_faithfulness": [],
    }

    for qa in track(samples, description="Evaluating..."):
        question = qa["question"]
        ground_truth = qa["answer"]
        context = qa.get("context", "")

        if rag_pipeline:
            answer, contexts = rag_pipeline(question)
        else:
            answer = ground_truth
            contexts = [context]

        faith = evaluate_faithfulness(answer, contexts)
        precision = evaluate_context_precision(question, contexts, ground_truth)
        recall = evaluate_context_recall(question, contexts, ground_truth)
        relevancy = evaluate_answer_relevancy(question, answer)
        correctness = evaluate_answer_correctness(question, answer, ground_truth)
        hallucination = evaluate_hallucination_rate(answer, contexts)
        translation = evaluate_translation_faithfulness(answer, contexts)

        results["faithfulness"].append(faith)
        results["context_precision"].append(precision)
        results["context_recall"].append(recall)
        results["answer_relevancy"].append(relevancy)
        results["answer_correctness"].append(correctness)
        results["hallucination_rate"].append(hallucination)
        results["translation_faithfulness"].append(translation)

        time.sleep(1)

    summary = {}
    for metric, scores in results.items():
        if scores:
            summary[metric] = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "count": len(scores),
            }

    table = Table(title="📊 Cross-lingual ArXiv RAG Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", style="green")
    table.add_column("Min", style="yellow")
    table.add_column("Max", style="green")
    table.add_column("Desired", style="dim")

    desired = {
        "faithfulness": "≥ 0.90",
        "context_precision": "≥ 0.80",
        "context_recall": "≥ 0.85",
        "answer_relevancy": "≥ 0.85",
        "answer_correctness": "≥ 0.80",
        "hallucination_rate": "≤ 0.10",
        "translation_faithfulness": "≥ 0.85",
    }

    for metric, stats in summary.items():
        table.add_row(
            metric.replace("_", " ").title(),
            f"{stats['mean']:.4f}",
            f"{stats['min']:.4f}",
            f"{stats['max']:.4f}",
            desired.get(metric, ""),
        )

    console.print(table)

    save_path = SYNTHETIC_DATA_DIR / "evaluation_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    console.print(f"[green]💾 Results saved: {save_path}[/]")

    return summary


if __name__ == "__main__":
    results = run_evaluation()
