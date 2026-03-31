"""
Sinh dữ liệu tổng hợp (Synthetic Data) cho đánh giá hệ thống Cross-lingual ArXiv RAG.
Sử dụng LLM để tạo cặp câu hỏi (tiếng Việt) - đáp án từ papers tiếng Anh.
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


GENERATE_QA_PROMPT = """Based on the following technical paper excerpt, generate {n_questions} question-answer pairs.

IMPORTANT RULES:
- Questions MUST be in VIETNAMESE (to test cross-lingual retrieval)
- Answers should reference the paper content accurately
- Questions should be diverse: concept questions, comparison questions, methodology questions
- Answer in JSON format

Paper excerpt:
{context}

Return JSON format:
[
    {{
        "question": "Câu hỏi tiếng Việt?",
        "answer": "Đáp án dựa trên paper...",
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
    """Sinh cặp Q&A cross-lingual từ paper excerpt."""
    client = get_client()
    prompt = GENERATE_QA_PROMPT.format(n_questions=n_questions, context=context)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You generate cross-lingual test data for an AI research RAG system. "
                                   "Questions in Vietnamese, answers based on English papers. Always reply in JSON."
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=2048,
            )

            raw = response.choices[0].message.content.strip()

            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                qa_pairs = json.loads(raw[start:end])
            else:
                qa_pairs = json.loads(raw)

            for pair in qa_pairs:
                pair["context"] = context

            return qa_pairs

        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            console.print(f"[red]❌ Cannot parse JSON from LLM[/]")
            return []
        except Exception as e:
            if "rate_limit" in str(e).lower():
                time.sleep(3 * (attempt + 1))
                continue
            console.print(f"[red]❌ Error generating Q&A: {e}[/]")
            return []


def generate_synthetic_dataset(
    documents: list[dict],
    questions_per_doc: int = 3,
    max_docs: int = 100,
    save: bool = True,
) -> list[dict]:
    """Sinh bộ synthetic data cross-lingual từ ArXiv papers."""
    console.print(f"[bold cyan]🔬 Generating cross-lingual synthetic Q&A data...[/]")

    all_pairs = []
    docs_to_process = documents[:max_docs]

    for doc in track(docs_to_process, description="Generating Q&A..."):
        text = doc.get("text", "")
        if len(text) < 100:
            continue

        context = text[:2000]
        pairs = generate_qa_pairs(context, n_questions=questions_per_doc)
        all_pairs.extend(pairs)

        time.sleep(2)

    console.print(f"[green]✅ Generated {len(all_pairs)} cross-lingual Q&A pairs[/]")

    if save and all_pairs:
        save_path = SYNTHETIC_DATA_DIR / "synthetic_qa.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_pairs, f, ensure_ascii=False, indent=2)
        console.print(f"[green]💾 Saved: {save_path}[/]")

    return all_pairs


if __name__ == "__main__":
    test_context = """
    We propose a new network architecture, the Transformer, based solely on
    attention mechanisms. Experiments on two machine translation tasks show
    these models to be superior in quality while being more parallelizable
    and requiring significantly less time to train. Our model achieves
    28.4 BLEU on the WMT 2014 English-to-German translation task.
    """

    pairs = generate_qa_pairs(test_context, n_questions=2)
    for p in pairs:
        console.print(f"\n[bold]Q:[/] {p['question']}")
        console.print(f"[dim]A: {p['answer']}[/]")
