"""
Script 02: Xây dựng cây RAPTOR từ dữ liệu đã tải.
Pipeline: Load → Clean → Chunk → Build RAPTOR Tree
"""
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data_processing.loader import load_local_data
from data_processing.cleaner import clean_documents
from data_processing.chunker import chunk_documents
from raptor.tree_builder import RAPTORTree
from config import RAPTOR_MAX_LEVELS
from rich.console import Console

console = Console()


def main(validate_only: bool = False):
    console.print("[bold]=" * 60)
    console.print("[bold cyan]🌳 BƯỚC 2: Xây dựng cây RAPTOR[/]")
    console.print("[bold]=" * 60)
    
    # 1. Load dữ liệu
    console.print("\n[bold]1/4 - Tải dữ liệu local...[/]")
    documents = load_local_data()
    
    # 2. Làm sạch
    console.print("\n[bold]2/4 - Làm sạch văn bản...[/]")
    cleaned = clean_documents(documents)
    
    # 3. Chunking
    console.print("\n[bold]3/4 - Phân mảnh văn bản...[/]")
    chunks = chunk_documents(cleaned)
    
    if validate_only:
        console.print(f"\n[green]✅ Validation OK: {len(chunks)} chunks sẵn sàng[/]")
        return
    
    # 4. Build RAPTOR tree
    console.print(f"\n[bold]4/4 - Xây dựng cây RAPTOR (max {RAPTOR_MAX_LEVELS} levels)...[/]")
    chunk_dicts = [c.to_dict() for c in chunks]
    
    tree = RAPTORTree(max_levels=RAPTOR_MAX_LEVELS)
    tree.build(chunk_dicts)
    tree.save()
    tree.visualize()
    
    console.print(f"\n[bold green]✅ Hoàn tất! Cây RAPTOR đã được lưu.[/]")
    console.print("[dim]Tiếp theo: chạy 03_index_qdrant.py[/]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true", help="Chỉ validate, không build")
    args = parser.parse_args()
    
    main(validate_only=args.validate_only)
