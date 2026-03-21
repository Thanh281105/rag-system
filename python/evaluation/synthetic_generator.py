"""
Sinh dữ liệu tổng hợp (Synthetic Data) cho đánh giá hệ thống RAG.
Sử dụng LLM để tự động tạo cặp câu hỏi - đáp án từ văn bản pháp lý.
"""
import json
import time
from groq import Groq
from rich.console import Console
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


GENERATE_QA_PROMPT = """Dựa trên đoạn văn bản pháp lý sau, hãy tạo ra {n_questions} cặp câu hỏi và đáp án.

Yêu cầu:
- Câu hỏi phải liên quan trực tiếp đến nội dung đoạn văn
- Câu hỏi phải đa dạng: có hỏi khái niệm, có hỏi điều kiện, có hỏi quy trình
- Đáp án phải chính xác và có trích dẫn từ đoạn văn
- Trả lời bằng JSON format

Đoạn văn bản pháp lý:
{context}

Trả lời theo JSON format:
[
    {{
        "question": "Câu hỏi 1?",
        "answer": "Đáp án 1...",
        "difficulty": "easy|medium|hard"
    }},
    ...
]

JSON:"""


def generate_qa_pairs(
    context: str,
    n_questions: int = 3,
    max_retries: int = 3,
) -> list[dict]:
    """
    Sinh cặp câu hỏi - đáp án từ đoạn văn bản pháp lý.
    
    Args:
        context: Đoạn văn bản pháp lý nguồn
        n_questions: Số cặp Q&A cần sinh
        max_retries: Số lần thử lại
        
    Returns:
        List[dict] với keys: question, answer, difficulty, context
    """
    client = get_client()
    prompt = GENERATE_QA_PROMPT.format(n_questions=n_questions, context=context)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn tạo dữ liệu kiểm thử cho hệ thống RAG pháp lý. Luôn trả lời bằng JSON."
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=2048,
            )
            
            raw = response.choices[0].message.content.strip()
            
            # Parse JSON
            # Tìm phần JSON trong response
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                qa_pairs = json.loads(raw[start:end])
            else:
                qa_pairs = json.loads(raw)
            
            # Thêm context vào mỗi pair
            for pair in qa_pairs:
                pair["context"] = context
            
            return qa_pairs
            
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            console.print(f"[red]❌ Không parse được JSON từ LLM[/]")
            return []
        except Exception as e:
            if "rate_limit" in str(e).lower():
                time.sleep(3 * (attempt + 1))
                continue
            console.print(f"[red]❌ Lỗi generate Q&A: {e}[/]")
            return []


def generate_synthetic_dataset(
    documents: list[dict],
    questions_per_doc: int = 3,
    max_docs: int = 100,
    save: bool = True,
) -> list[dict]:
    """
    Sinh bộ dữ liệu tổng hợp từ danh sách tài liệu.
    
    Args:
        documents: Danh sách tài liệu (với key 'text')
        questions_per_doc: Số câu hỏi mỗi tài liệu
        max_docs: Giới hạn số tài liệu xử lý
        save: Lưu ra file
        
    Returns:
        List[dict] - Bộ dữ liệu tổng hợp
    """
    console.print(f"[bold cyan]🔬 Sinh dữ liệu tổng hợp...[/]")
    
    all_pairs = []
    docs_to_process = documents[:max_docs]
    
    for doc in track(docs_to_process, description="Generating Q&A..."):
        text = doc.get("text", "")
        if len(text) < 100:
            continue
        
        # Lấy đoạn đại diện (tối đa 2000 chars)
        context = text[:2000]
        
        pairs = generate_qa_pairs(context, n_questions=questions_per_doc)
        all_pairs.extend(pairs)
        
        # Rate limiting cho Groq free tier
        time.sleep(2)
    
    console.print(f"[green]✅ Đã sinh {len(all_pairs)} cặp Q&A[/]")
    
    if save and all_pairs:
        save_path = SYNTHETIC_DATA_DIR / "synthetic_qa.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_pairs, f, ensure_ascii=False, indent=2)
        console.print(f"[green]💾 Đã lưu: {save_path}[/]")
    
    return all_pairs


if __name__ == "__main__":
    test_context = """
    Điều 111. Công ty cổ phần
    1. Công ty cổ phần là doanh nghiệp, trong đó:
    a) Vốn điều lệ được chia thành nhiều phần bằng nhau gọi là cổ phần;
    b) Cổ đông có thể là tổ chức, cá nhân; số lượng cổ đông tối thiểu là 03 và không hạn chế số lượng tối đa;
    c) Cổ đông chỉ chịu trách nhiệm về các khoản nợ và nghĩa vụ tài sản khác của doanh nghiệp trong phạm vi số vốn đã góp;
    d) Cổ đông có quyền tự do chuyển nhượng cổ phần của mình cho người khác.
    """
    
    pairs = generate_qa_pairs(test_context, n_questions=2)
    for p in pairs:
        console.print(f"\n[bold]Q:[/] {p['question']}")
        console.print(f"[dim]A: {p['answer']}[/]")
