"""
Script 05: Migrate dữ liệu từ RAPTOR tree sang LightRAG.

Pipeline:
1. Đọc dữ liệu RAPTOR tree hiện có (raptor_tree.json)
2. Extract text từ tất cả leaf nodes (level 0)
3. Insert vào LightRAG (build knowledge graph)
4. Verify bằng sample query

Chạy một lần duy nhất khi chuyển từ RAPTOR sang LightRAG.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import track

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from lightrag_setup.rag_instance import get_rag_instance, insert_document

console = Console()


async def migrate_from_raptor():
    """Migrate leaf nodes từ RAPTOR tree sang LightRAG."""
    console.print("[bold]==" * 30)
    console.print("[bold cyan]🔄 MIGRATION: RAPTOR → LightRAG[/]")
    console.print("[bold]==" * 30)

    # ─── Step 1: Load RAPTOR tree ────────────────────────
    raptor_path = PROCESSED_DATA_DIR / "raptor_tree.json"

    if not raptor_path.exists():
        console.print("[yellow]⚠️ Không tìm thấy raptor_tree.json[/]")
        console.print("[cyan]Thử tải từ raw data (arxiv_papers.json)...[/]")
        return await migrate_from_raw_data()

    with open(raptor_path, "r", encoding="utf-8") as f:
        tree_data = json.load(f)

    nodes = tree_data.get("nodes", {})
    total_nodes = len(nodes)
    console.print(f"[dim]📂 RAPTOR tree loaded: {total_nodes} nodes[/]")

    # Lọc leaf nodes (level 0 — chunks gốc, chứa text real)
    leaf_nodes = [
        node for node in nodes.values()
        if node.get("level", 0) == 0
    ]
    console.print(f"[dim]🍃 Leaf nodes (level 0): {len(leaf_nodes)}[/]")

    # ─── Step 2: Init LightRAG ───────────────────────────
    console.print("\n[cyan]🔄 Initializing LightRAG...[/]")
    rag = await get_rag_instance()
    console.print("[green]✅ LightRAG ready[/]")

    # ─── Step 3: Insert leaf nodes ───────────────────────
    console.print(f"\n[cyan]📥 Inserting {len(leaf_nodes)} documents into LightRAG...[/]")
    console.print("[dim]  (Using Ollama qwen2.5:3b for entity extraction)[/]")
    console.print("[dim]  ⚠️ This may take a while...[/]")

    success_count = 0
    fail_count = 0
    start_time = time.time()

    for i, node in enumerate(leaf_nodes):
        text = node.get("text", "")
        metadata = node.get("metadata", {})

        if not text or len(text) < 50:
            continue

        # Format with context
        doc_title = metadata.get("doc_title", "")
        authors = metadata.get("authors", "")
        year = metadata.get("year", "")
        arxiv_id = metadata.get("arxiv_id", "")

        contextualized = (
            f"[Paper: {doc_title} | Authors: {authors} | "
            f"Year: {year} | arXiv: {arxiv_id}]\n\n{text}"
        )

        try:
            ok = await insert_document(contextualized)
            if ok:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            console.print(f"[red]  ❌ Error inserting node #{node.get('node_id')}: {e}[/]")
            fail_count += 1

        # Progress log
        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(leaf_nodes) - i - 1) / rate if rate > 0 else 0
            console.print(
                f"[dim]  Progress: {i+1}/{len(leaf_nodes)} | "
                f"✅ {success_count} | ❌ {fail_count} | "
                f"ETA: {int(remaining)}s[/]"
            )

    elapsed = time.time() - start_time
    console.print(f"\n[bold green]🎉 Migration complete![/]")
    console.print(f"  ✅ Success: {success_count}")
    console.print(f"  ❌ Failed: {fail_count}")
    console.print(f"  ⏱️ Time: {elapsed:.1f}s")

    # ─── Step 4: Verify ──────────────────────────────────
    await verify_migration()


async def migrate_from_raw_data():
    """Fallback: migrate trực tiếp từ arxiv_papers.json."""
    raw_path = RAW_DATA_DIR / "arxiv_papers.json"

    if not raw_path.exists():
        console.print("[red]❌ Không tìm thấy dữ liệu. Hãy chạy 01_download_data.py trước.[/]")
        return

    with open(raw_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    console.print(f"[dim]📂 Loaded {len(papers)} papers from raw data[/]")

    # Chunk each paper then insert
    from data_processing.cleaner import clean_text
    from data_processing.chunker import semantic_chunk

    console.print("[cyan]🔄 Initializing LightRAG...[/]")
    rag = await get_rag_instance()

    success_count = 0
    start_time = time.time()

    for paper in papers:
        title = paper.get("title", "")
        text = paper.get("content", paper.get("text", ""))
        authors = paper.get("authors", "")
        year = paper.get("year", "")
        arxiv_id = paper.get("arxiv_id", "")

        if not text or len(text) < 100:
            continue

        cleaned = clean_text(text)
        chunks = semantic_chunk(
            cleaned,
            doc_id=paper.get("id", 0),
            doc_title=title,
            metadata={"authors": authors, "year": year, "arxiv_id": arxiv_id},
        )

        for chunk in chunks:
            contextualized = (
                f"[Paper: {title} | Authors: {authors} | "
                f"Year: {year} | arXiv: {arxiv_id}]\n\n{chunk['text']}"
            )
            try:
                await insert_document(contextualized)
                success_count += 1
            except Exception as e:
                console.print(f"[red]  ❌ Error: {e}[/]")

        console.print(f"[dim]  Processed: {title[:50]}... ({len(chunks)} chunks)[/]")

    elapsed = time.time() - start_time
    console.print(f"\n[bold green]🎉 Migration complete! {success_count} chunks in {elapsed:.1f}s[/]")
    await verify_migration()


async def verify_migration():
    """Verify bằng sample query."""
    console.print("\n[cyan]🔍 Verifying migration with sample query...[/]")

    from lightrag_setup.rag_instance import query_rag

    test_queries = [
        "What is the Transformer architecture?",
        "Explain attention mechanism in deep learning",
    ]

    for q in test_queries:
        result = await query_rag(q, mode="hybrid")
        preview = result[:200] if result else "(empty)"
        console.print(f"[dim]  Q: {q}[/]")
        console.print(f"[dim]  A: {preview}...[/]\n")

    console.print("[green]✅ Verification done! Check the answers above.[/]")


if __name__ == "__main__":
    asyncio.run(migrate_from_raptor())
