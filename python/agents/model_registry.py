"""
Shared Model Registry — Singleton cho tất cả AI models.

Tránh load model nhiều lần (BGE-M3 ~2.3GB, Reranker ~1GB).
Trên P1000 4GB VRAM, load 2 lần = tràn bộ nhớ = chết.
"""
from rich.console import Console

from utils.console import console

_embed_model = None
_reranker_model = None


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _use_fp16(model, enabled: bool, device: str, attr: str | None = None):
    if not enabled or not device.startswith("cuda"):
        return

    try:
        target = getattr(model, attr) if attr else model
        target.half()
    except Exception as exc:
        console.print(f"[yellow]⚠️ FP16 disabled for this model: {exc}[/]")


def get_embed_model():
    """Singleton embedding model — shared giữa router, query_engine, indexer."""
    global _embed_model
    if _embed_model is None:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))

        from sentence_transformers import SentenceTransformer
        from config import (
            EMBEDDING_DEVICE,
            EMBEDDING_FP16,
            EMBEDDING_MAX_SEQ_LENGTH,
            EMBEDDING_MODEL,
        )

        device = _resolve_device(EMBEDDING_DEVICE)
        console.print(
            f"[cyan]🔄 Loading embedding model: {EMBEDDING_MODEL} "
            f"(device={device})...[/]"
        )
        _embed_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        _embed_model.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH
        _use_fp16(_embed_model, EMBEDDING_FP16, device)
        console.print(
            f"[green]✅ Embedding model ready "
            f"({_embed_model.get_sentence_embedding_dimension()}D)[/]"
        )
    return _embed_model


def get_reranker_model():
    """Singleton reranker model (CrossEncoder)."""
    global _reranker_model
    if _reranker_model is None:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))

        from sentence_transformers import CrossEncoder
        from config import (
            RERANKER_DEVICE,
            RERANKER_FP16,
            RERANKER_MAX_LENGTH,
            RERANKER_MODEL,
        )

        device = _resolve_device(RERANKER_DEVICE)
        console.print(
            f"[cyan]🔄 Loading reranker model: {RERANKER_MODEL} "
            f"(device={device})...[/]"
        )
        _reranker_model = CrossEncoder(
            RERANKER_MODEL,
            max_length=RERANKER_MAX_LENGTH,
            device=device,
        )
        _use_fp16(_reranker_model, RERANKER_FP16, device, attr="model")
        console.print("[green]✅ Reranker model ready[/]")
    return _reranker_model


def warmup():
    """Pre-load tất cả models lúc startup thay vì lúc query đầu tiên."""
    from config import USE_RERANKER

    console.print("[bold cyan]🔥 Warming up models...[/]")
    get_embed_model()
    if USE_RERANKER:
        get_reranker_model()
    else:
        console.print("[dim]  Reranker disabled for low-VRAM mode[/]")
    console.print("[bold green]✅ All models ready![/]")
