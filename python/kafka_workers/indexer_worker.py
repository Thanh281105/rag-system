"""
LlamaIndex Indexer Worker — Kafka Consumer cho topic "document.chunks".

Thay thế lightrag_worker.py. Pipeline:
1. Nhận chunk từ Kafka topic "document.chunks"
2. Embed chunk bằng BGE-M3 (KHÔNG cần LLM)
3. Upsert vào Qdrant (dense + sparse vectors)
4. Publish trạng thái vào "document.indexed"

Tốc độ: ~30 papers trong 2-3 phút (vs 7-20 giờ với LightRAG).
"""
import signal
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from confluent_kafka import Consumer, Producer, KafkaError
from rich.console import Console

from kafka_workers.kafka_config import (
    BROKERS, GROUP_INDEXER,
    TOPIC_DOCUMENT_CHUNKS, TOPIC_DOCUMENT_INDEXED,
    serialize, deserialize,
)
from indexing.llamaindex_indexer import index_chunk

console = Console()
running = True


def signal_handler(sig, frame):
    global running
    console.print("[yellow]🛑 Shutting down indexer worker...[/]")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def create_consumer() -> Consumer:
    return Consumer({
        "bootstrap.servers": BROKERS,
        "group.id": GROUP_INDEXER,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
        "max.poll.interval.ms": 300000,  # 5 min (embedding nhanh hơn LLM rất nhiều)
    })


def create_producer() -> Producer:
    return Producer({
        "bootstrap.servers": BROKERS,
        "acks": "all",
    })


def run_worker():
    """Main loop cho LlamaIndex indexer worker."""
    console.print("[bold cyan]🚀 Starting LlamaIndex Indexer Worker...[/]")
    console.print(f"[dim]  Listening on: {TOPIC_DOCUMENT_CHUNKS}[/]")
    console.print(f"[dim]  Publishing to: {TOPIC_DOCUMENT_INDEXED}[/]")
    console.print("[dim]  Using BGE-M3 embedding only (NO LLM needed) ⚡[/]")

    consumer = create_consumer()
    producer = create_producer()
    consumer.subscribe([TOPIC_DOCUMENT_CHUNKS])

    # Stats
    indexed_count = 0
    failed_count = 0
    paper_chunk_tracker = {}

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
            chunk_event = deserialize(msg.value())
            paper_id = chunk_event.get("paper_id", 0)
            total_chunks = chunk_event.get("total_chunks", 1)

            # Index chunk (embedding-only, synchronous)
            success = index_chunk(chunk_event)

            if success:
                indexed_count += 1
                consumer.commit(msg)

                chunk_id = chunk_event.get("chunk_id", "?")
                console.print(
                    f"[green]  ✅ Indexed chunk {chunk_id}/{total_chunks} "
                    f"from '{chunk_event.get('doc_title', '')[:40]}...'[/]"
                )

                # Track paper completion
                if paper_id not in paper_chunk_tracker:
                    paper_chunk_tracker[paper_id] = {"total": total_chunks, "done": 0}
                paper_chunk_tracker[paper_id]["done"] += 1

                tracker = paper_chunk_tracker[paper_id]
                if tracker["done"] >= tracker["total"]:
                    producer.produce(
                        TOPIC_DOCUMENT_INDEXED,
                        key=str(paper_id).encode("utf-8"),
                        value=serialize({
                            "paper_id": paper_id,
                            "arxiv_id": chunk_event.get("arxiv_id", ""),
                            "status": "completed",
                            "total_chunks": total_chunks,
                            "doc_title": chunk_event.get("doc_title", ""),
                        }),
                    )
                    producer.flush()
                    console.print(
                        f"[bold green]📊 Paper fully indexed: "
                        f"'{chunk_event.get('doc_title', '')[:50]}' "
                        f"({total_chunks} chunks)[/]"
                    )
                    del paper_chunk_tracker[paper_id]
            else:
                failed_count += 1

            if (indexed_count + failed_count) % 10 == 0:
                console.print(
                    f"[dim]📈 Progress: {indexed_count} indexed, "
                    f"{failed_count} failed[/]"
                )

        except Exception as e:
            console.print(f"[red]❌ Indexing error: {e}[/]")
            import traceback
            traceback.print_exc()
            failed_count += 1

    consumer.close()
    producer.flush()
    console.print(
        f"[yellow]👋 Indexer stopped. "
        f"Total: {indexed_count} indexed, {failed_count} failed[/]"
    )


if __name__ == "__main__":
    run_worker()
