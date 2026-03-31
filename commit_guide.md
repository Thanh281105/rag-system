# Hướng dẫn Commit (Phase 2)

Bạn có thể dùng lệnh commit ngắn gọn này để lưu trữ tiến độ về chiến lược dữ liệu mới:

```bash
git add .
git commit -m "feat(rag): Implement Advanced Data Funnel & Strict Prompting

- **Data Loader**: Added year extraction, VBHN priority, and version-based deduplication across multiple dataset splits. Output format mapped to JSONL (content, source, version, category).
- **Format Injection**: Prefixed chunks precisely with `[{source} {version}] Điều X:` to retain explicit context for LLM.
- **Qdrant & Metadata**: Expanded Rust `LegalDocument` schema and Python Qdrant payloads to embed `version`, `source`, and `category` into vector payloads.
- **Analyst Prompt**: Set strict boundary conditions forcing explicit citations and automatic resolution of conflicting penal codes based on issue year."
```
