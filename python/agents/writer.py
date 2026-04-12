"""
Writer Agent — Groq Streaming answer generation.

Sinh câu trả lời tiếng Việt từ evidence tiếng Anh.
Hỗ trợ streaming (yield từng token) để đạt UX như ChatGPT.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from typing import AsyncGenerator, Callable, Awaitable, Optional

from agents.llm_client import (
    groq_complete,
    groq_stream_complete,
    GROQ_MODEL_SMART,
    GROQ_MODEL_FAST,
)


ANALYST_SYSTEM_PROMPT = """\
Bạn là trợ lý nghiên cứu AI chuyên nghiệp.
Nhiệm vụ: Trả lời câu hỏi tiếng Việt dựa TRỰC TIẾP và DUY NHẤT trên "Evidence" được cung cấp.

QUY TẮC BẮT BUỘC:
1. Chỉ sử dụng thông tin có trong Evidence. KHÔNG ĐƯỢC bịa đặt hay suy diễn ngoài evidence.
2. Giữ nguyên thuật ngữ kỹ thuật bằng tiếng Anh (Transformer, attention, LoRA, fine-tuning, etc.)
3. Trích dẫn theo format: Theo "[Tên Paper]" (Author, Year), ...
4. Nếu Evidence không đủ để trả lời → nói rõ "Tôi không tìm thấy thông tin này trong cơ sở dữ liệu"
5. Câu trả lời phải bằng TIẾNG VIỆT, rõ ràng và có cấu trúc

YÊU CẦU VỀ ĐỘ CHI TIẾT (BẮT BUỘC):
- PHẢI trích dẫn SỐ LIỆU CỤ THỂ nếu Evidence có (accuracy, F1, BLEU, parameters, latency, etc.)
- PHẢI mô tả PHƯƠNG PHÁP (methodology) mà paper sử dụng
- PHẢI giải thích TẠI SAO đằng sau kết quả
- KHÔNG ĐƯỢC chỉ paraphrase tiêu đề paper rồi kết luận chung chung
- Nếu Evidence chứa bảng so sánh, thí nghiệm → tóm tắt kết quả chính

CẤU TRÚC CÂU TRẢ LỜI:
1. Tóm tắt ngắn (1-2 câu trả lời trực tiếp)
2. Chi tiết phương pháp và kết quả (với số liệu cụ thể)
3. Kết luận/Hạn chế nếu Evidence có đề cập

SELF-CHECK:
- Các con số có khớp CHÍNH XÁC với Evidence không?
- Tên model/method có đúng không?
- Kết luận có được support bởi Evidence không?
"""

CASUAL_SYSTEM_PROMPT = (
    "Bạn là trợ lý nghiên cứu AI thân thiện. "
    "Trả lời ngắn gọn, lịch sự bằng tiếng Việt. "
    "Nếu người dùng hỏi về AI/ML, khuyên họ hỏi cụ thể hơn."
)


def _build_user_prompt(question_vn: str, evidence_text: str) -> str:
    """Build user prompt cho analyst."""
    if not evidence_text or len(evidence_text) < 50:
        return (
            f"CÂU HỎI (tiếng Việt): {question_vn}\n\n"
            f"EVIDENCE: Không tìm thấy thông tin liên quan.\n\n"
            f"Hãy thông báo cho người dùng biết rằng không có evidence."
        )

    return (
        f"CÂU HỎI (tiếng Việt): {question_vn}\n\n"
        f"EVIDENCE (từ Hybrid Search — tiếng Anh):\n{evidence_text[:6000]}\n\n"
        f"Hãy phân tích và trả lời câu hỏi bằng TIẾNG VIỆT dựa trên evidence trên."
    )


async def generate_answer(question_vn: str, evidence_text: str) -> str:
    """
    Sinh câu trả lời (non-streaming).

    Args:
        question_vn: Câu hỏi tiếng Việt
        evidence_text: Evidence text đã format

    Returns:
        Câu trả lời đầy đủ
    """
    user_prompt = _build_user_prompt(question_vn, evidence_text)

    return await groq_complete(
        prompt=user_prompt,
        system_prompt=ANALYST_SYSTEM_PROMPT,
        model=GROQ_MODEL_SMART,
        max_tokens=2048,
        temperature=0.1,
    )


async def generate_streaming(
    question_vn: str,
    evidence_text: str,
    stream_callback: Optional[Callable[[str], Awaitable[None]]] = None,
) -> str:
    """
    Sinh câu trả lời + streaming từng token.

    Args:
        question_vn: Câu hỏi tiếng Việt
        evidence_text: Evidence text đã format
        stream_callback: Hàm async được gọi mỗi khi có token mới

    Returns:
        Câu trả lời đầy đủ (cumulative)
    """
    user_prompt = _build_user_prompt(question_vn, evidence_text)

    full_answer = ""
    token_buffer = ""
    BUFFER_SIZE = 3  # Gom 3 tokens rồi mới gửi (giảm overhead Kafka)

    async for token in groq_stream_complete(
        prompt=user_prompt,
        system_prompt=ANALYST_SYSTEM_PROMPT,
        model=GROQ_MODEL_SMART,
        max_tokens=2048,
        temperature=0.1,
    ):
        full_answer += token
        token_buffer += token

        # Flush buffer khi đủ size hoặc gặp newline
        if len(token_buffer) >= BUFFER_SIZE or "\n" in token_buffer:
            if stream_callback:
                await stream_callback(token_buffer)
            token_buffer = ""

    # Flush remaining buffer
    if token_buffer and stream_callback:
        await stream_callback(token_buffer)

    return full_answer


async def generate_casual(question: str) -> str:
    """Trả lời casual (không cần RAG). Dùng model nhỏ cho tốc độ."""
    return await groq_complete(
        prompt=question,
        system_prompt=CASUAL_SYSTEM_PROMPT,
        model=GROQ_MODEL_FAST,
        max_tokens=256,
        temperature=0.7,
    )
