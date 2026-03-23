"""
Evaluator module: Đánh giá hệ thống RAG bằng các metrics chuẩn.
Metrics đầy đủ cho domain pháp lý:
- Context Precision: ngữ cảnh truy xuất có liên quan không
- Context Recall: ngữ cảnh có đủ để suy ra đáp án chuẩn không
- Faithfulness: câu trả lời có trung thực với ngữ cảnh không
- Answer Relevancy: câu trả lời có phù hợp với câu hỏi không
- Answer Correctness: câu trả lời có khớp với ground truth không
- Hallucination Rate: tỷ lệ thông tin bịa đặt (đặc biệt quan trọng cho pháp lý)
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
        # Parse số từ response (xử lý trường hợp LLM trả thêm text)
        import re
        match = re.search(r'(\d+\.?\d*)', score_str)
        if match:
            score = float(match.group(1))
            return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
        return 0.0
    except (ValueError, Exception):
        return 0.0


def evaluate_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Đánh giá Faithfulness: câu trả lời có đúng với ngữ cảnh hay không.
    Score 0-1, 1 = hoàn toàn trung thực.
    """
    context_str = "\n\n".join(contexts)
    prompt = f"""Đánh giá mức độ trung thực (faithfulness) của câu trả lời so với ngữ cảnh.

Ngữ cảnh: {context_str[:3000]}

Câu trả lời: {answer}

Cho điểm từ 0.0 đến 1.0:
- 1.0: Mọi thông tin đều có trong ngữ cảnh
- 0.5: Một phần thông tin đúng, một phần bị hallucinate
- 0.0: Hoàn toàn bịa đặt

CHỈ trả lời một số thập phân, ví dụ: 0.85"""
    return _llm_score(prompt)


def evaluate_context_precision(
    question: str,
    contexts: list[str],
    ground_truth: str,
) -> float:
    """
    Đánh giá Context Precision: ngữ cảnh truy xuất có liên quan không.
    Score 0-1, 1 = tất cả context đều liên quan.
    """
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
    return _llm_score(prompt)


def evaluate_context_recall(
    question: str,
    contexts: list[str],
    ground_truth: str,
) -> float:
    """
    Đánh giá Context Recall: ngữ cảnh có đủ thông tin để suy ra đáp án chuẩn không.
    Đây là metric quan trọng nhất cho legal RAG - đo xem retrieval có kéo đúng
    Điều luật cần thiết hay không.
    
    Score 0-1, 1 = toàn bộ đáp án có thể suy ra từ context.
    """
    context_str = "\n\n".join(contexts)
    prompt = f"""Đánh giá mức độ đầy đủ (recall) của ngữ cảnh truy xuất.

Câu hỏi: {question}
Đáp án chuẩn: {ground_truth[:1500]}

Ngữ cảnh truy xuất:
{context_str[:3000]}

Cho điểm từ 0.0 đến 1.0:
- 1.0: Toàn bộ thông tin trong đáp án chuẩn đều CÓ THỂ suy ra từ ngữ cảnh
- 0.5: Chỉ một phần đáp án có thể suy ra từ ngữ cảnh
- 0.0: Ngữ cảnh không chứa thông tin nào liên quan đến đáp án

CHỈ trả lời một số thập phân:"""
    return _llm_score(prompt)


def evaluate_answer_relevancy(question: str, answer: str) -> float:
    """
    Đánh giá Answer Relevancy: câu trả lời có trả lời đúng câu hỏi không.
    """
    prompt = f"""Đánh giá mức độ phù hợp của câu trả lời với câu hỏi.

Câu hỏi: {question}
Câu trả lời: {answer}

Cho điểm từ 0.0 đến 1.0:
- 1.0: Trả lời chính xác, đầy đủ
- 0.5: Trả lời một phần
- 0.0: Không liên quan

CHỈ trả lời một số thập phân:"""
    return _llm_score(prompt)


def evaluate_answer_correctness(
    question: str,
    answer: str,
    ground_truth: str,
) -> float:
    """
    Đánh giá Answer Correctness: câu trả lời có khớp với ground truth không.
    So sánh nội dung thực tế, không yêu cầu từ vựng giống hệt.
    """
    prompt = f"""So sánh câu trả lời với đáp án chuẩn và đánh giá mức độ chính xác.

Câu hỏi: {question}
Đáp án chuẩn: {ground_truth[:1500]}
Câu trả lời cần đánh giá: {answer[:1500]}

Cho điểm từ 0.0 đến 1.0:
- 1.0: Câu trả lời và đáp án chuẩn nêu cùng thông tin, cùng kết luận
- 0.5: Câu trả lời đúng một phần nhưng thiếu hoặc sai một phần
- 0.0: Câu trả lời hoàn toàn sai so với đáp án chuẩn

CHỈ trả lời một số thập phân:"""
    return _llm_score(prompt)


def evaluate_hallucination_rate(
    answer: str,
    contexts: list[str],
) -> float:
    """
    Đánh giá Hallucination Rate: tỷ lệ thông tin bịa đặt trong câu trả lời.
    Đặc biệt quan trọng cho pháp lý - kiểm tra:
    - Trích dẫn sai số Điều/Khoản
    - Bịa tên luật/nghị định không tồn tại
    - Thêm thông tin không có trong bằng chứng
    
    Score 0-1, 0 = không hallucination (TỐT), 1 = toàn hallucination (XẤU).
    """
    context_str = "\n\n".join(contexts)
    prompt = f"""Phân tích tỷ lệ hallucination (thông tin bịa đặt) trong câu trả lời pháp lý.

Bằng chứng (ngữ cảnh gốc):
{context_str[:3000]}

Câu trả lời cần kiểm tra:
{answer[:1500]}

Kiểm tra đặc biệt:
- Số Điều, Khoản, Điểm có đúng với ngữ cảnh không?
- Tên luật, nghị định, thông tư có tồn tại trong ngữ cảnh không?
- Có thông tin nào KHÔNG xuất hiện trong ngữ cảnh không?

Cho điểm TỶ LỆ HALLUCINATION từ 0.0 đến 1.0:
- 0.0: Không có hallucination nào (TỐT)
- 0.3: Có một vài chi tiết nhỏ bị bịa
- 0.5: Khoảng nửa thông tin bị bịa
- 1.0: Hoàn toàn bịa đặt (XẤU)

CHỈ trả lời một số thập phân:"""
    return _llm_score(prompt)


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
    console.print(f"[bold cyan]📊 Đánh giá {len(samples)} mẫu (6 metrics)...[/]")
    
    results = {
        "faithfulness": [],
        "context_precision": [],
        "context_recall": [],
        "answer_relevancy": [],
        "answer_correctness": [],
        "hallucination_rate": [],
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
        
        # Đánh giá 6 metrics
        faith = evaluate_faithfulness(answer, contexts)
        precision = evaluate_context_precision(question, contexts, ground_truth)
        recall = evaluate_context_recall(question, contexts, ground_truth)
        relevancy = evaluate_answer_relevancy(question, answer)
        correctness = evaluate_answer_correctness(question, answer, ground_truth)
        hallucination = evaluate_hallucination_rate(answer, contexts)
        
        results["faithfulness"].append(faith)
        results["context_precision"].append(precision)
        results["context_recall"].append(recall)
        results["answer_relevancy"].append(relevancy)
        results["answer_correctness"].append(correctness)
        results["hallucination_rate"].append(hallucination)
        
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
    table = Table(title="📊 RAG Evaluation Results (Legal Domain)")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", style="green")
    table.add_column("Min", style="yellow")
    table.add_column("Max", style="green")
    table.add_column("Desired", style="dim")
    
    # Mục tiêu cho domain pháp lý
    desired = {
        "faithfulness": "≥ 0.90",
        "context_precision": "≥ 0.80",
        "context_recall": "≥ 0.85",
        "answer_relevancy": "≥ 0.85",
        "answer_correctness": "≥ 0.80",
        "hallucination_rate": "≤ 0.10",
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
    
    # Lưu kết quả
    save_path = SYNTHETIC_DATA_DIR / "evaluation_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    console.print(f"[green]💾 Kết quả đã lưu: {save_path}[/]")
    
    return summary


if __name__ == "__main__":
    results = run_evaluation()
