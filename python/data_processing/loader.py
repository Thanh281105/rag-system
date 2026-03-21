"""
Tải bộ dữ liệu UTS_VLC (Vietnamese Legal Corpus) từ HuggingFace.
Dataset chứa các văn bản luật và nghị định Việt Nam.
"""
import json
from pathlib import Path
from datasets import load_dataset
from rich.console import Console
from rich.progress import track

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR

console = Console()


def download_dataset(save_dir: Path = RAW_DATA_DIR) -> list[dict]:
    """
    Tải dataset undertheseanlp/UTS_VLC từ HuggingFace và lưu local.
    
    Returns:
        List[dict]: Danh sách các tài liệu pháp lý, mỗi tài liệu là dict
                    với các trường: title, text, metadata...
    """
    # Lấy luật mới nhất (split 2026) theo yêu cầu
    console.print("[cyan]Đang tải phần 2026 (Luật mới nhất)...[/]")
    dataset = load_dataset("undertheseanlp/UTS_VLC", split="2026")
    
    console.print(f"[green]✅ Đã tải {len(dataset)} tài liệu pháp lý mới nhất[/]")
    console.print(f"[dim]Các cột: {dataset.column_names}[/]")
    
    # Lưu ra file JSON để dùng offline
    save_path = save_dir / "uts_vlc_raw.json"
    documents = []
    
    for idx, item in enumerate(track(dataset, description="Đang lưu dữ liệu...")):
        doc = {
            "id": idx,
            "title": item.get("title", ""),
            "text": item.get("text", item.get("content", "")),
            "metadata": {
                k: v for k, v in item.items() 
                if k not in ("title", "text", "content")
            }
        }
        documents.append(doc)
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    console.print(f"[green]💾 Đã lưu vào {save_path}[/]")
    
    return documents


def load_local_data(data_path: Path = None) -> list[dict]:
    """Đọc dữ liệu đã tải về từ file JSON local."""
    if data_path is None:
        data_path = RAW_DATA_DIR / "uts_vlc_raw.json"
    
    if not data_path.exists():
        console.print("[yellow]⚠️ Chưa có dữ liệu local. Đang tải từ HuggingFace...[/]")
        return download_dataset()
    
    with open(data_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
    
    console.print(f"[green]📂 Đã đọc {len(documents)} tài liệu từ {data_path}[/]")
    return documents


if __name__ == "__main__":
    docs = download_dataset()
    console.print(f"\n[bold]Tổng số tài liệu: {len(docs)}[/]")
    if docs:
        console.print(f"[dim]Ví dụ tiêu đề: {docs[0].get('title', 'N/A')}[/]")
        text_preview = docs[0].get("text", "")[:200]
        console.print(f"[dim]Nội dung (200 ký tự đầu): {text_preview}...[/]")
