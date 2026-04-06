"""
Ingestion Worker — Kafka Consumer cho topic "paper.uploaded".

Pipeline:
1. Nhận event paper.uploaded (chứa arxiv_id, metadata)
2. Download PDF → Extract text
3. Clean + Chunk text
4. Publish từng chunk vào topic "document.chunks"
"""
import asyncio
import signal
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from confluent_kafka import Consumer, Producer, KafkaError
from rich.console import Console

from kafka_workers.kafka_config import (
    BROKERS, GROUP_INGESTION,
    TOPIC_PAPER_UPLOADED, TOPIC_DOCUMENT_CHUNKS,
    serialize, deserialize,
)
from data_processing.loader import download_arxiv_papers, extract_text_from_pdf
from data_processing.cleaner import clean_document
from data_processing.chunker import chunk_documents
from config import RAW_DATA_DIR

console = Console()
running = True


def signal_handler(sig, frame):
    global running
    console.print("[yellow]🛑 Shutting down ingestion worker...[/]")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def create_consumer() -> Consumer:
    """Tạo Kafka consumer cho ingestion."""
    return Consumer({
        "bootstrap.servers": BROKERS,
        "group.id": GROUP_INGESTION,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
        "auto.commit.interval.ms": 5000,
    })


def create_producer() -> Producer:
    """Tạo Kafka producer để publish chunks."""
    return Producer({
        "bootstrap.servers": BROKERS,
        "acks": "all",
    })


def process_paper(paper_event: dict, producer: Producer):
    """
    Xử lý một paper: download → clean → chunk → publish chunks.
    """
    arxiv_id = paper_event.get("arxiv_id", "")
    title = paper_event.get("title", "Unknown")
    paper_id = paper_event.get("paper_id", 0)

    console.print(f"[cyan]📄 Processing paper: {title} (arXiv:{arxiv_id})[/]")

    # Lấy text từ event (nếu đã có) hoặc extract từ PDF
    text = paper_event.get("text", "")

    if not text:
        # Thử đọc từ PDF local
        pdf_dir = RAW_DATA_DIR / "pdfs"
        pdf_path = pdf_dir / f"{arxiv_id.replace('/', '_')}.pdf"
        if pdf_path.exists():
            text = extract_text_from_pdf(pdf_path)
        else:
            console.print(f"[red]❌ No text/PDF found for {arxiv_id}[/]")
            return

    if len(text) < 100:
        console.print(f"[yellow]⚠️ Text too short ({len(text)} chars), skipping[/]")
        return

    # Clean text
    cleaned_text = clean_document(text)

    # Chunk text
    chunks = chunk_documents([{
        "id": paper_id,
        "title": title,
        "text": cleaned_text,
        "authors": paper_event.get("authors", ""),
        "year": paper_event.get("year", 0),
        "arxiv_id": arxiv_id,
    }])

    console.print(f"[dim]  → {len(chunks)} chunks created[/]")

    # Publish từng chunk vào Kafka
    for i, chunk in enumerate(chunks):
        chunk_event = {
            "paper_id": paper_id,
            "chunk_id": i,
            "text": chunk.text,
            "doc_title": title,
            "authors": paper_event.get("authors", ""),
            "year": paper_event.get("year", 0),
            "arxiv_id": arxiv_id,
            "total_chunks": len(chunks),
        }
        producer.produce(
            TOPIC_DOCUMENT_CHUNKS,
            key=f"{arxiv_id}:{i}".encode("utf-8"),
            value=serialize(chunk_event),
        )

    producer.flush()
    console.print(f"[green]✅ Published {len(chunks)} chunks for '{title}'[/]")


def run_worker():
    """Main loop cho ingestion worker."""
    console.print("[bold cyan]🚀 Starting Ingestion Worker...[/]")
    console.print(f"[dim]  Listening on: {TOPIC_PAPER_UPLOADED}[/]")
    console.print(f"[dim]  Publishing to: {TOPIC_DOCUMENT_CHUNKS}[/]")

    consumer = create_consumer()
    producer = create_producer()

    consumer.subscribe([TOPIC_PAPER_UPLOADED])

    while running:
        msg = consumer.poll(timeout=1.0)

        if msg is None:
            continue

        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            console.print(f"[red]❌ Kafka error: {msg.error()}[/]")
            continue

        try:
            event = deserialize(msg.value())
            process_paper(event, producer)
        except Exception as e:
            console.print(f"[red]❌ Processing error: {e}[/]")
            import traceback
            traceback.print_exc()

    consumer.close()
    producer.flush()
    console.print("[yellow]👋 Ingestion worker stopped.[/]")


if __name__ == "__main__":
    run_worker()
