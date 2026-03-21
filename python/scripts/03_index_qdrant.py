"""
Script 03: Đưa cây RAPTOR vào Qdrant vector database.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from raptor.tree_builder import RAPTORTree
from retrieval.qdrant_client import QdrantWrapper
from rich.console import Console

console = Console()


def main():
    console.print("[bold]=" * 60)
    console.print("[bold cyan]📦 BƯỚC 3: Index vào Qdrant[/]")
    console.print("[bold]=" * 60)
    
    # 1. Load cây RAPTOR
    console.print("\n[bold]1/3 - Tải cây RAPTOR...[/]")
    tree = RAPTORTree.load()
    
    # 2. Kết nối Qdrant & tạo collection
    console.print("\n[bold]2/3 - Kết nối Qdrant...[/]")
    qdrant = QdrantWrapper()
    qdrant.create_collection(recreate=True)
    
    # 3. Upsert tất cả nodes
    console.print("\n[bold]3/3 - Upsert nodes...[/]")
    nodes = tree.get_all_nodes()
    
    node_dicts = [n.to_dict() for n in nodes]
    embeddings = np.array([n.embedding for n in nodes])
    
    qdrant.upsert_nodes(node_dicts, embeddings)
    
    # Verify
    info = qdrant.get_collection_info()
    console.print(f"\n[bold green]✅ Hoàn tất! Collection info:[/]")
    for k, v in info.items():
        console.print(f"  {k}: {v}")
    
    console.print("[dim]Tiếp theo: khởi động Rust backend[/]")


if __name__ == "__main__":
    main()
