"""
Script 01: Tải dữ liệu pháp lý từ HuggingFace.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data_processing.loader import download_dataset
from rich.console import Console

console = Console()

if __name__ == "__main__":
    console.print("[bold]=" * 60)
    console.print("[bold cyan]📥 BƯỚC 1: Tải dữ liệu UTS_VLC[/]")
    console.print("[bold]=" * 60)
    
    docs = download_dataset()
    
    console.print(f"\n[bold green]✅ Hoàn tất! {len(docs)} tài liệu đã được tải.[/]")
    console.print("[dim]Tiếp theo: chạy 02_build_raptor.py[/]")
