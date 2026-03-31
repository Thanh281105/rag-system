"""
Tải và xử lý dữ liệu từ ArXiv API.
Pipeline:
1. Search papers theo topic (cs.AI)
2. Download PDF + extract metadata
3. Extract raw text từ PDF bằng pymupdf
4. Lưu ra JSON format chuẩn
"""
import json
import re
import time
from pathlib import Path
import arxiv
import fitz  # pymupdf
from rich.console import Console
from rich.progress import track

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR, ARXIV_TOPIC, ARXIV_MAX_PAPERS

console = Console()

# Thư mục tạm cho PDF
PDF_DIR = RAW_DATA_DIR / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)


def download_arxiv_papers(
    topic: str = ARXIV_TOPIC,
    max_papers: int = ARXIV_MAX_PAPERS,
    save_dir: Path = RAW_DATA_DIR,
) -> list[dict]:
    """
    Tải papers từ ArXiv API, extract text từ PDF.

    Args:
        topic: ArXiv category (e.g., "cs.AI")
        max_papers: Số papers tối đa
        save_dir: Thư mục lưu

    Returns:
        List[dict] với keys: id, title, authors, year, abstract, content, arxiv_id
    """
    console.print(f"[cyan]🔍 Tìm kiếm {max_papers} papers từ ArXiv ({topic})...[/]")

    # Search ArXiv
    search = arxiv.Search(
        query=f"cat:{topic}",
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    client = arxiv.Client(
        page_size=10,
        delay_seconds=3.0,
        num_retries=3,
    )

    documents = []
    idx = 0

    for paper in track(
        client.results(search),
        description="Downloading papers...",
        total=max_papers,
    ):
        try:
            arxiv_id = paper.entry_id.split("/abs/")[-1]
            year = paper.published.year
            authors = ", ".join([a.name for a in paper.authors[:5]])
            if len(paper.authors) > 5:
                authors += " et al."

            # Download PDF
            pdf_path = PDF_DIR / f"{arxiv_id.replace('/', '_')}.pdf"
            if not pdf_path.exists():
                paper.download_pdf(dirpath=str(PDF_DIR), filename=pdf_path.name)
                time.sleep(1)  # Rate limiting

            # Extract text từ PDF
            content = extract_text_from_pdf(pdf_path)

            if len(content) < 500:
                console.print(f"[yellow]⚠️ Bỏ qua {arxiv_id}: nội dung quá ngắn ({len(content)} chars)[/]")
                continue

            doc = {
                "id": idx,
                "title": paper.title,
                "authors": authors,
                "year": year,
                "abstract": paper.summary,
                "content": content,
                "arxiv_id": arxiv_id,
                "text": content,  # Alias cho compat với pipeline
            }
            documents.append(doc)
            idx += 1

            console.print(f"[dim]  ✅ [{idx}] {paper.title[:60]}... ({year})[/]")

        except Exception as e:
            console.print(f"[red]❌ Lỗi tải paper: {e}[/]")
            continue

    console.print(f"[bold green]✅ Đã tải {len(documents)} papers từ ArXiv[/]")

    # Save JSON
    save_path = save_dir / "arxiv_papers.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    console.print(f"[green]💾 Đã lưu vào {save_path}[/]")
    return documents


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract raw text từ PDF bằng pymupdf.
    Xử lý multi-column layout và loại bỏ noise cơ bản.
    """
    try:
        doc = fitz.open(str(pdf_path))
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
        doc.close()
        return "\n\n".join(pages)
    except Exception as e:
        console.print(f"[red]❌ Lỗi đọc PDF {pdf_path.name}: {e}[/]")
        return ""


def load_local_data(data_path: Path = None) -> list[dict]:
    """Đọc dữ liệu đã tải về từ file JSON local."""
    if data_path is None:
        data_path = RAW_DATA_DIR / "arxiv_papers.json"

    if not data_path.exists():
        console.print("[yellow]⚠️ Chưa có dữ liệu local. Đang tải từ ArXiv...[/]")
        return download_arxiv_papers()

    with open(data_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    console.print(f"[green]📂 Đã đọc {len(documents)} papers từ {data_path}[/]")
    return documents


if __name__ == "__main__":
    docs = download_arxiv_papers()
