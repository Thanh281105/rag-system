"""
Translator Agent — Dịch câu hỏi VN → EN.

Sử dụng Groq (model FAST) để dịch, giữ nguyên thuật ngữ kỹ thuật.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.llm_client import groq_complete, GROQ_MODEL_FAST


TRANSLATE_SYSTEM_PROMPT = (
    "You are a precise translator. Translate Vietnamese to English. "
    "Keep technical terms (RAG, Transformer, LoRA, RLHF, attention, etc.) unchanged. "
    "Only output the translation, nothing else."
)


async def translate_to_english(question_vn: str) -> str:
    """
    Dịch câu hỏi từ tiếng Việt sang tiếng Anh.

    Args:
        question_vn: Câu hỏi tiếng Việt

    Returns:
        Câu hỏi đã dịch sang tiếng Anh
    """
    prompt = (
        f"Translate the following Vietnamese question to English. "
        f"Keep all technical terms unchanged.\n\n"
        f"Vietnamese: {question_vn}\n\nEnglish:"
    )

    translated = await groq_complete(
        prompt=prompt,
        system_prompt=TRANSLATE_SYSTEM_PROMPT,
        model=GROQ_MODEL_FAST,
        max_tokens=256,
        temperature=0.0,
    )

    return translated.strip()
