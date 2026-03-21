"""
Xây dựng cây RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval).
Đệ quy: chunks gốc → embed → cluster → summarize → embed summaries → cluster → ... → root
"""
import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from rich.console import Console
from rich.tree import Tree as RichTree

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RAPTOR_MAX_LEVELS, PROCESSED_DATA_DIR
from raptor.embedder import embed_texts
from raptor.clustering import cluster_pipeline
from raptor.summarizer import summarize_clusters

console = Console()


@dataclass
class RAPTORNode:
    """Một node trong cây RAPTOR."""
    node_id: int
    text: str
    embedding: np.ndarray = None
    level: int = 0  # 0 = leaf (chunk gốc), 1+ = summary levels
    children_ids: list[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Serialize node (bỏ embedding vì lưu riêng)."""
        return {
            "node_id": self.node_id,
            "text": self.text,
            "level": self.level,
            "children_ids": self.children_ids,
            "metadata": self.metadata,
        }


class RAPTORTree:
    """
    Cây RAPTOR hoàn chỉnh.
    Xây dựng cây tri thức phân cấp từ dưới lên trên.
    """
    
    def __init__(self, max_levels: int = RAPTOR_MAX_LEVELS):
        self.max_levels = max_levels
        self.nodes: dict[int, RAPTORNode] = {}
        self.node_counter = 0
        self.levels: dict[int, list[int]] = {}  # level → node_ids
    
    def _next_id(self) -> int:
        """Generate node ID tuần tự."""
        nid = self.node_counter
        self.node_counter += 1
        return nid
    
    def build(self, chunks: list[dict]) -> "RAPTORTree":
        """
        Xây dựng cây RAPTOR từ danh sách chunks.
        
        Args:
            chunks: List[dict] với key 'text', 'doc_id', 'chunk_id', 'doc_title'
            
        Returns:
            self (fluent API)
        """
        console.print("[bold cyan]🌳 Bắt đầu xây dựng cây RAPTOR...[/]")
        
        # ─── Level 0: Leaf nodes (chunks gốc) ───────────────
        console.print(f"\n[bold]Level 0: {len(chunks)} chunks gốc[/]")
        
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        
        current_level_ids = []
        for i, chunk in enumerate(chunks):
            node = RAPTORNode(
                node_id=self._next_id(),
                text=chunk["text"],
                embedding=embeddings[i],
                level=0,
                metadata={
                    "doc_id": chunk.get("doc_id", 0),
                    "chunk_id": chunk.get("chunk_id", i),
                    "doc_title": chunk.get("doc_title", ""),
                }
            )
            self.nodes[node.node_id] = node
            current_level_ids.append(node.node_id)
        
        self.levels[0] = current_level_ids
        
        # ─── Level 1+: Đệ quy cluster → summarize ──────────
        for level in range(1, self.max_levels + 1):
            if len(current_level_ids) <= 3:
                console.print(
                    f"[yellow]⚠️ Dừng ở level {level-1}: "
                    f"chỉ còn {len(current_level_ids)} nodes[/]"
                )
                break
            
            console.print(f"\n[bold]Level {level}:[/]")
            
            # Lấy embeddings + texts của level hiện tại
            current_embeddings = np.array([
                self.nodes[nid].embedding for nid in current_level_ids
            ])
            current_texts = [
                self.nodes[nid].text for nid in current_level_ids
            ]
            
            # Cluster
            clusters = cluster_pipeline(current_embeddings)
            
            if len(clusters) <= 1 and level > 1:
                console.print("[yellow]⚠️ Chỉ còn 1 cụm duy nhất → dừng[/]")
                break
            
            # Summarize từng cluster
            cluster_texts = []
            cluster_child_ids = []
            for cluster_indices in clusters:
                c_texts = [current_texts[idx] for idx in cluster_indices]
                c_ids = [current_level_ids[idx] for idx in cluster_indices]
                cluster_texts.append(c_texts)
                cluster_child_ids.append(c_ids)
            
            summaries = summarize_clusters(
                clusters, current_texts, rate_limit_delay=1.5
            )
            
            # Embed summaries
            summary_embeddings = embed_texts(summaries, show_progress=False)
            
            # Tạo nodes mới cho level này
            new_level_ids = []
            for i, summary in enumerate(summaries):
                node = RAPTORNode(
                    node_id=self._next_id(),
                    text=summary,
                    embedding=summary_embeddings[i],
                    level=level,
                    children_ids=cluster_child_ids[i],
                    metadata={"cluster_size": len(cluster_child_ids[i])},
                )
                self.nodes[node.node_id] = node
                new_level_ids.append(node.node_id)
            
            self.levels[level] = new_level_ids
            current_level_ids = new_level_ids
            
            console.print(
                f"[green]  → Tạo {len(new_level_ids)} nodes tóm tắt[/]"
            )
        
        # Thống kê
        total = len(self.nodes)
        console.print(f"\n[bold green]🌳 Cây RAPTOR hoàn thành![/]")
        console.print(f"  Tổng nodes: {total}")
        for lvl, ids in sorted(self.levels.items()):
            console.print(f"  Level {lvl}: {len(ids)} nodes")
        
        return self
    
    def get_all_nodes(self) -> list[RAPTORNode]:
        """Lấy tất cả nodes (collapsed tree)."""
        return list(self.nodes.values())
    
    def get_nodes_by_level(self, level: int) -> list[RAPTORNode]:
        """Lấy nodes theo level."""
        ids = self.levels.get(level, [])
        return [self.nodes[nid] for nid in ids]
    
    def save(self, save_dir: Path = PROCESSED_DATA_DIR):
        """
        Lưu cây RAPTOR ra file.
        - Metadata → JSON
        - Embeddings → NPY
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Lưu metadata
        tree_data = {
            "total_nodes": len(self.nodes),
            "levels": {str(k): v for k, v in self.levels.items()},
            "nodes": {
                str(nid): node.to_dict() 
                for nid, node in self.nodes.items()
            }
        }
        
        meta_path = save_dir / "raptor_tree.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(tree_data, f, ensure_ascii=False, indent=2)
        
        # Lưu embeddings
        node_ids = sorted(self.nodes.keys())
        embeddings = np.array([self.nodes[nid].embedding for nid in node_ids])
        emb_path = save_dir / "raptor_embeddings.npy"
        np.save(emb_path, embeddings)
        
        # Lưu mapping node_id → embedding index
        id_map = {str(nid): idx for idx, nid in enumerate(node_ids)}
        map_path = save_dir / "raptor_id_map.json"
        with open(map_path, "w") as f:
            json.dump(id_map, f)
        
        console.print(f"[green]💾 Đã lưu cây RAPTOR: {meta_path}[/]")
        console.print(f"[green]💾 Embeddings: {emb_path} ({embeddings.shape})[/]")
    
    @classmethod
    def load(cls, load_dir: Path = PROCESSED_DATA_DIR) -> "RAPTORTree":
        """Tải cây RAPTOR từ file."""
        meta_path = load_dir / "raptor_tree.json"
        emb_path = load_dir / "raptor_embeddings.npy"
        map_path = load_dir / "raptor_id_map.json"
        
        with open(meta_path, "r", encoding="utf-8") as f:
            tree_data = json.load(f)
        
        embeddings = np.load(emb_path)
        
        with open(map_path, "r") as f:
            id_map = json.load(f)
        
        tree = cls()
        tree.levels = {int(k): v for k, v in tree_data["levels"].items()}
        
        for nid_str, node_data in tree_data["nodes"].items():
            nid = int(nid_str)
            emb_idx = id_map.get(nid_str, 0)
            
            node = RAPTORNode(
                node_id=nid,
                text=node_data["text"],
                embedding=embeddings[emb_idx],
                level=node_data["level"],
                children_ids=node_data.get("children_ids", []),
                metadata=node_data.get("metadata", {}),
            )
            tree.nodes[nid] = node
        
        tree.node_counter = max(tree.nodes.keys()) + 1 if tree.nodes else 0
        
        console.print(f"[green]📂 Đã tải cây RAPTOR: {len(tree.nodes)} nodes[/]")
        return tree
    
    def visualize(self, max_depth: int = 3):
        """Hiển thị cây RAPTOR dạng tree trong terminal."""
        rich_tree = RichTree("🌳 RAPTOR Tree")
        
        max_level = max(self.levels.keys()) if self.levels else 0
        
        for level in range(min(max_level + 1, max_depth)):
            level_branch = rich_tree.add(f"[bold]Level {level}[/] ({len(self.levels.get(level, []))} nodes)")
            for nid in self.levels.get(level, [])[:5]:  # Max 5 nodes per level
                node = self.nodes[nid]
                preview = node.text[:80].replace('\n', ' ')
                level_branch.add(f"[dim]#{nid}: {preview}...[/]")
            remaining = len(self.levels.get(level, [])) - 5
            if remaining > 0:
                level_branch.add(f"[dim]... và {remaining} nodes khác[/]")
        
        console.print(rich_tree)


if __name__ == "__main__":
    # Test với dữ liệu nhỏ
    test_chunks = [
        {"text": "Điều 1. Phạm vi điều chỉnh. Luật này quy định về thành lập doanh nghiệp.", "doc_id": 0, "chunk_id": 0, "doc_title": "Luật DN"},
        {"text": "Điều 2. Đối tượng áp dụng. Doanh nghiệp và cá nhân liên quan.", "doc_id": 0, "chunk_id": 1, "doc_title": "Luật DN"},
        {"text": "Điều 3. Giải thích từ ngữ. Doanh nghiệp là tổ chức có tên riêng.", "doc_id": 0, "chunk_id": 2, "doc_title": "Luật DN"},
        {"text": "Điều 4. Bảo đảm của Nhà nước đối với doanh nghiệp.", "doc_id": 0, "chunk_id": 3, "doc_title": "Luật DN"},
    ]
    
    tree = RAPTORTree(max_levels=2)
    tree.build(test_chunks)
    tree.visualize()
