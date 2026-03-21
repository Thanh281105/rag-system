"""
HyDE (Hypothetical Document Embeddings) - Sinh tài liệu giả định.
Khi nhận câu hỏi pháp lý, LLM sẽ sinh ra một câu trả lời nháp
(có cấu trúc và từ vựng pháp lý) để embedding tìm kiếm chính xác hơn.
"""
import numpy as np
from groq import Groq
from rich.console import Console

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import GROQ_API_KEY, LLM_MODEL, GROQ_TEMPERATURE
from raptor.embedder import embed_single

console = Console()

# Groq client
_client = None


def get_client() -> Groq:
    """Lazy init Groq client."""
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


HYDE_PROMPT = """Bạn là chuyên gia luật Việt Nam. Hãy viết một đoạn văn bản pháp lý ngắn gọn (150-200 từ) để trả lời câu hỏi sau. Viết như thể bạn đang trích dẫn từ bộ luật thật.

Yêu cầu:
- Sử dụng văn phong pháp lý chính thức
- Trích dẫn số điều, khoản nếu có thể (dù có thể không chính xác)
- Viết bằng tiếng Việt
- Không cần disclaimer về tính chính xác

Câu hỏi: {question}

VĂN BẢN PHÁP LÝ THAM KHẢO:"""


def generate_hypothetical_document(question: str) -> str:
    """
    Sinh tài liệu giả định từ câu hỏi pháp lý.
    
    Args:
        question: Câu hỏi pháp lý từ người dùng
        
    Returns:
        Văn bản giả định có cấu trúc pháp lý
    """
    client = get_client()
    
    prompt = HYDE_PROMPT.format(question=question)
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Bạn viết văn bản pháp lý giả định để hỗ trợ tìm kiếm."
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,  # Cao hơn bình thường để tạo đa dạng
            max_tokens=512,
        )
        
        hypothetical = response.choices[0].message.content.strip()
        console.print(f"[dim]📝 HyDE generated ({len(hypothetical)} chars)[/]")
        return hypothetical
        
    except Exception as e:
        console.print(f"[red]❌ HyDE error: {e}[/]")
        # Fallback: dùng câu hỏi gốc
        return question


def hyde_embed(question: str) -> np.ndarray:
    """
    Pipeline HyDE hoàn chỉnh:
    1. Sinh tài liệu giả định
    2. Embed tài liệu giả định
    
    Args:
        question: Câu hỏi pháp lý
        
    Returns:
        Vector embedding của tài liệu giả định
    """
    # Sinh tài liệu giả định
    hypothetical = generate_hypothetical_document(question)
    
    # Embed
    embedding = embed_single(hypothetical)
    
    return embedding


def hyde_embed_with_original(question: str, alpha: float = 0.5) -> np.ndarray:
    """
    Kết hợp embedding HyDE và embedding câu hỏi gốc.
    Tránh trường hợp HyDE hallucinate quá xa câu hỏi.
    
    Args:
        question: Câu hỏi pháp lý
        alpha: Trọng số cho HyDE (1-alpha cho câu hỏi gốc)
        
    Returns:
        Vector embedding kết hợp (đã normalize)
    """
    hyde_emb = hyde_embed(question)
    original_emb = embed_single(question)
    
    # Kết hợp weighted
    combined = alpha * hyde_emb + (1 - alpha) * original_emb
    
    # L2 normalize
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    
    return combined


if __name__ == "__main__":
    # Test HyDE
    question = "Điều kiện để thành lập doanh nghiệp tư nhân là gì?"
    
    console.print(f"[bold]Câu hỏi:[/] {question}\n")
    
    hypothetical = generate_hypothetical_document(question)
    console.print(f"[bold]Tài liệu giả định:[/]\n{hypothetical}\n")
    
    embedding = hyde_embed(question)
    console.print(f"[bold]Embedding shape:[/] {embedding.shape}")
