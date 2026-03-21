"""
Evaluator module: Đánh giá hệ thống RAG bằng các metrics chuẩn.
Đo lường Context Precision, Faithfulness, Answer Relevancy.
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


def evaluate_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Đánh giá Faithfulness: câu trả lời có đúng với ngữ cảnh hay không.
    Score 0-1, 1 = hoàn toàn trung thực.
    """
    client = get_client()
    
    context_str = "\n\n".join(contexts)
    prompt = f"""Đánh giá mức độ trung thực (faithfulness) của câu trả lời so với ngữ cảnh.

Ngữ cảnh: {context_str[:3000]}

Câu trả lời: {answer}

Cho điểm từ 0.0 đến 1.0:
- 1.0: Mọi thông tin đều có trong ngữ cảnh
- 0.5: Một phần thông tin đúng, một phần bị hallucinate
- 0.0: Hoàn toàn bịa đặt

CHỈ trả lời một số thập phân, ví dụ: 0.85"""
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        score_str = response.choices[0].message.content.strip()
        return float(score_str)
    except (ValueError, Exception):
        return 0.0


def evaluate_context_precision(
    question: str,
    contexts: list[str],
    ground_truth: str,
) -> float:
    """
    Đánh giá Context Precision: ngữ cảnh truy xuất có liên quan không.
    Score 0-1, 1 = tất cả context đều liên quan.
    """
    client = get_client()
    
    context_str = "\n\n".join(f"[Context {i+1}]: {c}" for i, c in enumerate(contexts))
    prompt = f"""Đánh giá precision của ngữ cảnh truy xuất cho câu hỏi.

Câu hỏi: {question}
Đáp án chuẩn: {ground_truth}

Ngữ cảnh truy xuất:
{context_str[:3000]}

Cho điểm từ 0.0 đến 1.0:
- 1.0: Tất cả context đều liên quan đến câu hỏi
- 0.5: Một nửa context liên quan
- 0.0: Không context nào liên quan

CHỈ trả lời một số thập phân:"""
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        return float(response.choices[0].message.content.strip())
    except (ValueError, Exception):
        return 0.0


def evaluate_answer_relevancy(question: str, answer: str) -> float:
    """
    Đánh giá Answer Relevancy: câu trả lời có trả lời đúng câu hỏi không.
    """
    client = get_client()
    
    prompt = f"""Đánh giá mức độ phù hợp của câu trả lời với câu hỏi.

Câu hỏi: {question}
Câu trả lời: {answer}

Cho điểm từ 0.0 đến 1.0:
- 1.0: Trả lời chính xác, đầy đủ
- 0.5: Trả lời một phần
- 0.0: Không liên quan

CHỈ trả lời một số thập phân:"""
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        return float(response.choices[0].message.content.strip())
    except (ValueError, Exception):
        return 0.0


def run_evaluation(
    qa_pairs: list[dict] = None,
    rag_pipeline: callable = None,
    max_samples: int = 50,
) -> dict:
    """
    Chạy đánh giá toàn diện trên bộ synthetic data.
    
    Args:
        qa_pairs: Bộ dữ liệu Q&A tổng hợp
        rag_pipeline: Function nhận question → trả về (answer, contexts)
        max_samples: Số mẫu đánh giá tối đa
        
    Returns:
        Dict chứa các metrics trung bình
    """
    # Load synthetic data nếu chưa có
    if qa_pairs is None:
        qa_path = SYNTHETIC_DATA_DIR / "synthetic_qa.json"
        if not qa_path.exists():
            console.print("[red]❌ Chưa có synthetic data. Chạy synthetic_generator trước.[/]")
            return {}
        with open(qa_path, "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)
    
    samples = qa_pairs[:max_samples]
    console.print(f"[bold cyan]📊 Đánh giá {len(samples)} mẫu...[/]")
    
    results = {
        "faithfulness": [],
        "context_precision": [],
        "answer_relevancy": [],
    }
    
    for qa in track(samples, description="Evaluating..."):
        question = qa["question"]
        ground_truth = qa["answer"]
        context = qa.get("context", "")
        
        # Chạy RAG pipeline
        if rag_pipeline:
            answer, contexts = rag_pipeline(question)
        else:
            # Dùng ground truth context nếu không có pipeline
            answer = ground_truth
            contexts = [context]
        
        # Đánh giá
        faith = evaluate_faithfulness(answer, contexts)
        precision = evaluate_context_precision(question, contexts, ground_truth)
        relevancy = evaluate_answer_relevancy(question, answer)
        
        results["faithfulness"].append(faith)
        results["context_precision"].append(precision)
        results["answer_relevancy"].append(relevancy)
        
        time.sleep(1)  # Rate limiting
    
    # Tính trung bình
    summary = {}
    for metric, scores in results.items():
        if scores:
            summary[metric] = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "count": len(scores),
            }
    
    # Hiển thị bảng kết quả
    table = Table(title="📊 RAG Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", style="green")
    table.add_column("Min", style="yellow")
    table.add_column("Max", style="green")
    
    for metric, stats in summary.items():
        table.add_row(
            metric.replace("_", " ").title(),
            f"{stats['mean']:.4f}",
            f"{stats['min']:.4f}",
            f"{stats['max']:.4f}",
        )
    
    console.print(table)
    
    # Lưu kết quả
    save_path = SYNTHETIC_DATA_DIR / "evaluation_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    console.print(f"[green]💾 Kết quả đã lưu: {save_path}[/]")
    
    return summary


if __name__ == "__main__":
    results = run_evaluation()
