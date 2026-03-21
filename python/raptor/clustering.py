"""
Phân cụm embedding vectors sử dụng UMAP (giảm chiều) + GMM (Gaussian Mixture Model).
Đây là bước quan trọng trong RAPTOR để nhóm các điều khoản liên quan.
"""
import numpy as np
from sklearn.mixture import GaussianMixture
from rich.console import Console

console = Console()


def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 10,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Giảm chiều embedding bằng UMAP để clustering tốt hơn.
    
    Args:
        embeddings: Ma trận embedding (n, d)
        n_components: Số chiều đầu ra
        n_neighbors: Số láng giềng cho UMAP
        min_dist: Khoảng cách tối thiểu
        metric: Metric distance
        
    Returns:
        Ma trận đã giảm chiều (n, n_components)
    """
    import umap
    
    # Điều chỉnh n_neighbors nếu dataset nhỏ
    actual_neighbors = max(2, min(n_neighbors, len(embeddings) - 1))
    actual_components = max(1, min(n_components, len(embeddings) - 2))
    
    reducer = umap.UMAP(
        n_components=actual_components,
        n_neighbors=actual_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    
    reduced = reducer.fit_transform(embeddings)
    console.print(
        f"[dim]UMAP: {embeddings.shape[1]}D → {reduced.shape[1]}D[/]"
    )
    
    return reduced


def cluster_embeddings(
    embeddings: np.ndarray,
    max_clusters: int = None,
    threshold: float = 0.5,
) -> tuple[list[list[int]], int]:
    """
    Phân cụm bằng GMM (soft clustering).
    
    Mỗi vector có thể thuộc nhiều cụm (soft assignment) → phù hợp cho
    văn bản pháp lý vì 1 điều khoản có thể liên quan đến nhiều chủ đề.
    
    Args:
        embeddings: Ma trận embedding đã giảm chiều
        max_clusters: Số cụm tối đa (auto nếu None)
        threshold: Ngưỡng xác suất để gán vào cụm
        
    Returns:
        (clusters, n_clusters):
            clusters: List[List[int]] - danh sách indices cho mỗi cụm
            n_clusters: Số cụm thực tế
    """
    n_samples = len(embeddings)
    
    if n_samples <= 1:
        return [[0]] if n_samples == 1 else [], n_samples
    
    # Tự động xác định số cụm tối ưu bằng BIC
    if max_clusters is None:
        max_clusters = min(n_samples // 2, 50)  # Tối đa 50 cụm
    max_clusters = max(max_clusters, 2)
    
    best_bic = float('inf')
    best_n = 2
    
    # Tìm số cụm tối ưu
    search_range = range(2, min(max_clusters + 1, n_samples))
    for n in search_range:
        try:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type="full",
                random_state=42,
                max_iter=200,
            )
            gmm.fit(embeddings)
            bic = gmm.bic(embeddings)
            if bic < best_bic:
                best_bic = bic
                best_n = n
        except Exception:
            continue
    
    # Fit final model
    gmm = GaussianMixture(
        n_components=best_n,
        covariance_type="full",
        random_state=42,
        max_iter=300,
    )
    gmm.fit(embeddings)
    
    # Soft assignment: xác suất thuộc mỗi cụm
    probs = gmm.predict_proba(embeddings)  # (n_samples, n_clusters)
    
    # Gán vào cụm dựa trên threshold
    clusters = [[] for _ in range(best_n)]
    for idx in range(n_samples):
        for cluster_id in range(best_n):
            if probs[idx, cluster_id] >= threshold:
                clusters[cluster_id].append(idx)
    
    # Loại bỏ cụm rỗng
    clusters = [c for c in clusters if len(c) > 0]
    
    console.print(
        f"[green]📊 GMM: phân thành {len(clusters)} cụm từ {n_samples} vectors "
        f"(BIC tối ưu ở k={best_n})[/]"
    )
    
    # Thống kê kích thước cụm
    sizes = [len(c) for c in clusters]
    if sizes:
        console.print(
            f"[dim]Kích thước cụm: min={min(sizes)}, max={max(sizes)}, "
            f"avg={sum(sizes)/len(sizes):.1f}[/]"
        )
    
    return clusters, len(clusters)


def cluster_pipeline(
    embeddings: np.ndarray,
    umap_dim: int = 10,
) -> list[list[int]]:
    """
    Pipeline hoàn chỉnh: UMAP + GMM.
    
    Args:
        embeddings: Ma trận embedding gốc
        umap_dim: Số chiều sau UMAP
        
    Returns:
        List[List[int]] - Danh sách clusters, mỗi cluster chứa indices
    """
    if len(embeddings) <= 3:
        # Quá ít để cluster → trả về 1 cụm duy nhất
        return [list(range(len(embeddings)))]
    
    # Bước 1: Giảm chiều
    reduced = reduce_dimensions(embeddings, n_components=umap_dim)
    
    # Bước 2: Phân cụm
    clusters, n_clusters = cluster_embeddings(reduced)
    
    return clusters


if __name__ == "__main__":
    # Test với dữ liệu ngẫu nhiên
    np.random.seed(42)
    
    # Tạo 3 cụm rõ ràng
    cluster1 = np.random.randn(20, 1024) + np.array([1] * 1024)
    cluster2 = np.random.randn(15, 1024) + np.array([-1] * 1024)
    cluster3 = np.random.randn(10, 1024) + np.array([0] * 1024)
    
    test_embeddings = np.vstack([cluster1, cluster2, cluster3])
    
    clusters = cluster_pipeline(test_embeddings)
    console.print(f"\n[bold]Kết quả:[/]")
    for i, c in enumerate(clusters):
        console.print(f"  Cụm {i}: {len(c)} phần tử")
