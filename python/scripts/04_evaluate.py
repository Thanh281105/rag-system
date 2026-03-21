"""
Script 04: Chạy đánh giá hệ thống RAG.
1. Sinh synthetic Q&A (nếu chưa có)
2. Chạy evaluation pipeline
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from evaluation.synthetic_generator import generate_synthetic_dataset
from evaluation.evaluator import run_evaluation
from data_processing.loader import load_local_data
from config import SYNTHETIC_DATA_DIR
from rich.console import Console

console = Console()


def main():
    console.print("[bold]=" * 60)
    console.print("[bold cyan]📊 BƯỚC 4: Đánh giá hệ thống RAG[/]")
    console.print("[bold]=" * 60)
    
    # 1. Sinh synthetic data nếu chưa có
    qa_path = SYNTHETIC_DATA_DIR / "synthetic_qa.json"
    if not qa_path.exists():
        console.print("\n[bold]1/2 - Sinh dữ liệu tổng hợp...[/]")
        documents = load_local_data()
        generate_synthetic_dataset(
            documents,
            questions_per_doc=3,
            max_docs=50,
        )
    else:
        console.print("[green]📂 Đã có synthetic data sẵn[/]")
    
    # 2. Chạy evaluation
    console.print("\n[bold]2/2 - Chạy đánh giá...[/]")
    results = run_evaluation(max_samples=30)
    
    console.print(f"\n[bold green]✅ Đánh giá hoàn tất![/]")


if __name__ == "__main__":
    main()
