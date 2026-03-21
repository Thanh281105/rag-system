"""
Tóm tắt cụm văn bản pháp lý bằng Groq API (LLaMA-3.3-70b).
Dùng trong RAPTOR để tạo node cha từ các node con đã phân cụm.
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


SUMMARIZE_PROMPT = """Bạn là chuyên gia pháp luật Việt Nam. Hãy tóm tắt ngắn gọn nội dung chính của các đoạn văn bản pháp lý dưới đây thành MỘT đoạn văn duy nhất.

Yêu cầu:
- Giữ nguyên các thuật ngữ pháp lý quan trọng (số hiệu luật, tên điều khoản...)
- Tóm tắt ngắn gọn nhưng đầy đủ ý chính
- Không thêm thông tin ngoài nội dung được cung cấp
- Viết bằng tiếng Việt
- Tối đa 300 từ

Các đoạn văn bản:

{texts}

TÓM TẮT:"""


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
    
    # Nối các đoạn với separator
    combined = "\n\n---\n\n".join(
        f"[Đoạn {i+1}]:\n{t}" for i, t in enumerate(texts)
    )
    
    # Giới hạn input length (Groq context window)
    if len(combined) > 12000:
        combined = combined[:12000] + "\n\n[...bị cắt do quá dài...]"
    
    prompt = SUMMARIZE_PROMPT.format(texts=combined)
    
    for attempt in range(max_retries):
        try:
            client = get_client(api_keys[current_key_idx])
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là trợ lý pháp lý chuyên tóm tắt văn bản luật Việt Nam."
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
                    console.print(f"[yellow]⚠️ Key {current_key_idx+1}/{len(api_keys)} hết hạn mức, chuyển sang key tiếp theo...[/]")
                    current_key_idx = (current_key_idx + 1) % len(api_keys)
                    
                    if current_key_idx == 0:
                        wait_time = retry_delay * (attempt + 1)
                        console.print(f"[yellow]⏳ Tất cả {len(api_keys)} keys đều quá tải - đợi {wait_time}s...[/]")
                        time.sleep(wait_time)
                else:
                    wait_time = retry_delay * (attempt + 1)
                    console.print(
                        f"[yellow]⏳ Rate limit - đợi {wait_time}s (lần {attempt+1}/{max_retries})[/]"
                    )
                    time.sleep(wait_time)
            else:
                console.print(f"[red]❌ Lỗi Groq API: {error_msg}[/]")
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
    
    Args:
        clusters: List[List[int]] - indices của từng cụm
        texts: List[str] - tất cả văn bản
        rate_limit_delay: Delay giữa các API calls (giây)
        
    Returns:
        List[str] - Tóm tắt cho mỗi cụm
    """
    summaries = []
    
    console.print(f"[cyan]📝 Đang tóm tắt {len(clusters)} cụm...[/]")
    
    for i, cluster_indices in enumerate(clusters):
        cluster_texts = [texts[idx] for idx in cluster_indices]
        
        console.print(
            f"[dim]  Cụm {i+1}/{len(clusters)}: "
            f"{len(cluster_texts)} đoạn văn bản[/]"
        )
        
        summary = summarize_cluster(cluster_texts)
        summaries.append(summary)
        
        # Rate limiting
        if i < len(clusters) - 1:
            time.sleep(rate_limit_delay)
    
    console.print(f"[green]✅ Đã tóm tắt {len(summaries)} cụm[/]")
    return summaries


if __name__ == "__main__":
    # Test tóm tắt
    test_texts = [
        "Điều 1. Luật này quy định về thành lập doanh nghiệp.",
        "Điều 2. Đối tượng áp dụng bao gồm doanh nghiệp và cá nhân liên quan.",
        "Điều 3. Trường hợp luật chuyên ngành có quy định đặc thù thì áp dụng luật đó.",
    ]
    
    summary = summarize_cluster(test_texts)
    console.print(f"\n[bold]Tóm tắt:[/]\n{summary}")
