"""
Qdrant client wrapper cho hệ thống RAG pháp lý.
Quản lý collection, upsert, và search trên Qdrant vector database.
"""
import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    SparseVectorParams, SparseIndexParams,
    NamedVector, NamedSparseVector, SparseVector,
    SearchRequest, Filter, FieldCondition, MatchValue,
    models,
)
from rich.console import Console

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, EMBEDDING_DIM

console = Console()


class QdrantWrapper:
    """Wrapper cho Qdrant operations."""
    
    def __init__(
        self,
        url: str = QDRANT_URL,
        api_key: str = QDRANT_API_KEY,
        collection_name: str = QDRANT_COLLECTION,
    ):
        self.collection_name = collection_name
        
        kwargs = {"url": url, "timeout": 60}
        if api_key:
            kwargs["api_key"] = api_key
        
        self.client = QdrantClient(**kwargs)
        console.print(f"[green]✅ Kết nối Qdrant: {url}[/]")
    
    def create_collection(self, recreate: bool = False):
        """
        Tạo collection với hỗ trợ Hybrid Search (Dense + Sparse vectors).
        
        Args:
            recreate: Nếu True, xóa collection cũ và tạo mới
        """
        exists = self.client.collection_exists(self.collection_name)
        
        if exists and not recreate:
            info = self.client.get_collection(self.collection_name)
            console.print(
                f"[yellow]📦 Collection '{self.collection_name}' đã tồn tại "
                f"({info.points_count} points)[/]"
            )
            return
        
        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            console.print(f"[yellow]🗑️ Đã xóa collection cũ[/]")
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        
        console.print(
            f"[green]✅ Đã tạo collection '{self.collection_name}' "
            f"(dense: {EMBEDDING_DIM}D + sparse BM25)[/]"
        )
    
    def upsert_nodes(
        self,
        nodes: list[dict],
        embeddings: np.ndarray,
        batch_size: int = 100,
    ):
        """
        Thêm RAPTOR nodes vào Qdrant.
        
        Args:
            nodes: List[dict] với metadata
            embeddings: Dense embeddings
            batch_size: Kích thước batch
        """
        total = len(nodes)
        console.print(f"[cyan]📤 Đang upsert {total} nodes vào Qdrant...[/]")
        
        for i in range(0, total, batch_size):
            batch_nodes = nodes[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            points = []
            for j, (node, emb) in enumerate(zip(batch_nodes, batch_embeddings)):
                # Tạo sparse vector đơn giản từ text (BM25-like)
                text = node.get("text", "")
                sparse_indices, sparse_values = self._text_to_sparse(text)
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": emb.tolist(),
                        "sparse": SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                    },
                    payload={
                        "text": text,
                        "node_id": node.get("node_id", i + j),
                        "level": node.get("level", 0),
                        "doc_title": node.get("doc_title", ""),
                        "doc_id": node.get("doc_id", 0),
                        "metadata": node.get("metadata", {}),
                    },
                )
                points.append(point)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
        
        console.print(f"[green]✅ Đã upsert {total} nodes thành công[/]")
    
    def _text_to_sparse(self, text: str) -> tuple[list[int], list[float]]:
        """
        Tạo sparse vector đơn giản từ text (word frequency based).
        Sử dụng hash để map từ → index.
        """
        words = text.lower().split()
        word_freq = {}
        for w in words:
            h = hash(w) % 100000  # Map to fixed sparse dim
            word_freq[h] = word_freq.get(h, 0) + 1
        
        indices = list(word_freq.keys())
        values = [float(v) for v in word_freq.values()]
        
        return indices, values
    
    def search_dense(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        level_filter: int = None,
    ) -> list[dict]:
        """
        Tìm kiếm Dense vector (semantic search).
        
        Args:
            query_vector: Vector truy vấn
            top_k: Số kết quả
            level_filter: Lọc theo RAPTOR level (None = tất cả)
            
        Returns:
            List[dict] kết quả
        """
        query_filter = None
        if level_filter is not None:
            query_filter = Filter(
                must=[FieldCondition(key="level", match=MatchValue(value=level_filter))]
            )
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            using="dense",
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        
        return [
            {
                "id": str(r.id),
                "score": r.score,
                "text": r.payload.get("text", ""),
                "level": r.payload.get("level", 0),
                "doc_title": r.payload.get("doc_title", ""),
                "node_id": r.payload.get("node_id", 0),
            }
            for r in results.points
        ]
    
    def search_sparse(
        self,
        query_text: str,
        top_k: int = 20,
    ) -> list[dict]:
        """
        Tìm kiếm Sparse vector (keyword/BM25-like search).
        
        Args:
            query_text: Câu truy vấn text
            top_k: Số kết quả
            
        Returns:
            List[dict] kết quả
        """
        indices, values = self._text_to_sparse(query_text)
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=SparseVector(indices=indices, values=values),
            using="sparse",
            limit=top_k,
            with_payload=True,
        )
        
        return [
            {
                "id": str(r.id),
                "score": r.score,
                "text": r.payload.get("text", ""),
                "level": r.payload.get("level", 0),
                "doc_title": r.payload.get("doc_title", ""),
                "node_id": r.payload.get("node_id", 0),
            }
            for r in results.points
        ]
    
    def get_collection_info(self) -> dict:
        """Lấy thông tin collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status.value,
            }
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    # Test kết nối
    wrapper = QdrantWrapper()
    info = wrapper.get_collection_info()
    console.print(f"[bold]Collection info:[/] {info}")
