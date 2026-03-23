# Hướng dẫn Commit

Bạn có thể dùng đoạn nội dung sau để commit tất cả các thay đổi vừa rồi. Nó tóm tắt đầy đủ kỹ thuật nhưng vẫn ngắn gọn.

```bash
git add .
git commit -m "feat(rag): Upgrade pipeline for VN legal domain

- **Chunking**: Implement hierarchy-aware chunking (Phần->...->Khoản->Điểm) with auto parent context injection. Switched to tiktoken for exact overlap.
- **Reranker**: Replaced LLM scoring with 'bge-reranker-v2-m3' cross-encoder microservice for 5x-10x speedup in Rust backend.
- **Sparse Vector (BM25)**: Added Vietnamese stop-words removal and log-scaled TF to heavily boost keyword matching.
- **Data Cleaner**: Added intensive regex patterns to strip OCR watermarks, footnotes, and signature blocks from gov documents.
- **Evaluation**: Integrated Context Recall, Answer Correctness, and Hallucination Rate metrics.
- **Config**: Extracted RRF weights into .env for easy A/B tuning."
```
