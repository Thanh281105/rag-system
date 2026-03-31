"""
Tóm tắt cụm văn bản kỹ thuật bằng Groq API (LLaMA-3.3-70b).
Dùng trong RAPTOR để tạo node cha từ các node con đã phân cụm.
Summaries bằng TIẾNG ANH (vì corpus là EN).
"""
import time
from groq import Groq
from rich.console import Console

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import GROQ_API_KEY, LLM_MODEL, GROQ_MAX_TOKENS, GROQ_TEMPERATURE

console = Console()

# Groq clients
_clients = {}

def get_client(api_key: str = None) -> Groq:
    """Lazy init Groq client with specific key."""
    global _clients
    import config

    if not api_key:
        api_key = config.GROQ_API_KEYS[0] if config.GROQ_API_KEYS else None

    if not api_key:
        raise ValueError(
            "❌ GROQ_API_KEYS chưa được cấu hình! "
            "Vui lòng tạo API key tại https://console.groq.com "
            "và thêm vào file .env"
        )

    if api_key not in _clients:
        _clients[api_key] = Groq(api_key=api_key)

    return _clients[api_key]


SUMMARIZE_PROMPT = """You are an expert AI research paper summarizer. Summarize the following text passages into ONE concise paragraph in English.

Requirements:
- Keep all technical terms, model names, and important numbers
- Preserve key findings and methodological details
- Do NOT add information not present in the source
- Maximum 300 words

Text passages:

{texts}

SUMMARY:"""


def summarize_cluster(
    texts: list[str],
    max_retries: int = 15,
    retry_delay: float = 2.0,
) -> str:
    """
    Tóm tắt một cụm văn bản bằng Groq API (hỗ trợ xoay vòng API Keys).
    """
    import config
    api_keys = config.GROQ_API_KEYS if config.GROQ_API_KEYS else [config.GROQ_API_KEY]
    current_key_idx = 0

    combined = "\n\n---\n\n".join(
        f"[Passage {i+1}]:\n{t}" for i, t in enumerate(texts)
    )

    if len(combined) > 12000:
        combined = combined[:12000] + "\n\n[...truncated...]"

    prompt = SUMMARIZE_PROMPT.format(texts=combined)

    for attempt in range(max_retries):
        try:
            client = get_client(api_keys[current_key_idx])
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a technical research paper summarizer. Always respond in English."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=GROQ_TEMPERATURE,
                max_tokens=GROQ_MAX_TOKENS,
            )

            summary = response.choices[0].message.content.strip()
            return summary

        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg or "quota" in error_msg.lower():
                if len(api_keys) > 1:
                    console.print(f"[yellow]⚠️ Key {current_key_idx+1}/{len(api_keys)} rate limited, switching...[/]")
                    current_key_idx = (current_key_idx + 1) % len(api_keys)
                    if current_key_idx == 0:
                        wait_time = retry_delay * (attempt + 1)
                        console.print(f"[yellow]⏳ All keys exhausted - waiting {wait_time}s...[/]")
                        time.sleep(wait_time)
                else:
                    wait_time = retry_delay * (attempt + 1)
                    console.print(f"[yellow]⏳ Rate limit - waiting {wait_time}s ({attempt+1}/{max_retries})[/]")
                    time.sleep(wait_time)
            else:
                console.print(f"[red]❌ Groq API error: {error_msg}[/]")
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay)

    return ""


def summarize_clusters(
    clusters: list[list[int]],
    texts: list[str],
    rate_limit_delay: float = 1.0,
) -> list[str]:
    """
    Tóm tắt tất cả các cụm.
    """
    summaries = []

    console.print(f"[cyan]📝 Summarizing {len(clusters)} clusters...[/]")

    for i, cluster_indices in enumerate(clusters):
        cluster_texts = [texts[idx] for idx in cluster_indices]

        console.print(
            f"[dim]  Cluster {i+1}/{len(clusters)}: "
            f"{len(cluster_texts)} passages[/]"
        )

        summary = summarize_cluster(cluster_texts)
        summaries.append(summary)

        if i < len(clusters) - 1:
            time.sleep(rate_limit_delay)

    console.print(f"[green]✅ Summarized {len(summaries)} clusters[/]")
    return summaries


if __name__ == "__main__":
    test_texts = [
        "The Transformer architecture uses multi-head self-attention mechanisms.",
        "BERT is a pre-trained model that uses bidirectional transformers for NLP tasks.",
        "GPT-3 demonstrates that large language models can perform few-shot learning.",
    ]

    summary = summarize_cluster(test_texts)
    console.print(f"\n[bold]Summary:[/]\n{summary}")
