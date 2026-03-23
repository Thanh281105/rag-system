"""
Phân mảnh văn bản pháp lý thành các đoạn (chunks) phù hợp cho embedding.
Hỗ trợ chunking theo cấu trúc pháp lý (Điều, Khoản, Chương).
"""
import re
import tiktoken
from dataclasses import dataclass, field
from rich.console import Console

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP

console = Console()

# Tokenizer để đếm tokens chính xác
_ENCODER = tiktoken.get_encoding("cl100k_base")


@dataclass
class TextChunk:
    """Một mảnh văn bản với metadata."""
    text: str
    doc_id: int
    chunk_id: int
    doc_title: str = ""
    level: int = 0  # 0 = leaf node (chunk gốc)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "doc_title": self.doc_title,
            "level": self.level,
            "metadata": self.metadata,
        }


def count_tokens(text: str) -> int:
    """Đếm số tokens trong text."""
    return len(_ENCODER.encode(text))


def split_by_legal_structure(text: str) -> list[str]:
    """
    Chia văn bản theo cấu trúc pháp lý (Điều, Chương, Mục).
    Ưu tiên giữ nguyên cấu trúc logic của luật.
    """
    # Pattern nhận diện Điều, Chương, Mục trong luật VN
    patterns = [
        r'(?=\n\s*(?:Điều\s+\d+))',          # Chia theo Điều
        r'(?=\n\s*(?:Chương\s+[IVXLCDM]+))',  # Chia theo Chương  
        r'(?=\n\s*(?:Mục\s+\d+))',            # Chia theo Mục
    ]
    
    # Thử chia từ đơn vị nhỏ nhất (Điều) trước
    for pattern in patterns:
        parts = re.split(pattern, text)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            return parts
    
    # Nếu không tìm thấy cấu trúc → chia theo đoạn văn
    parts = text.split('\n\n')
    return [p.strip() for p in parts if p.strip()]


def chunk_with_overlap(
    sections: list[str],
    max_tokens: int = CHUNK_SIZE,
    overlap_tokens: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Gộp các section nhỏ hoặc chia section lớn, đảm bảo overlap.
    
    Args:
        sections: Danh sách các đoạn văn bản
        max_tokens: Số tokens tối đa mỗi chunk
        overlap_tokens: Số tokens overlap giữa các chunk
    
    Returns:
        Danh sách chunks đã xử lý
    """
    chunks = []
    current_chunk = ""
    
    for section in sections:
        section_tokens = count_tokens(section)
        current_tokens = count_tokens(current_chunk)
        
        if section_tokens > max_tokens:
            # Section quá dài → chia nhỏ theo câu
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            sentences = re.split(r'(?<=[.!?])\s+', section)
            temp_chunk = ""
            
            for sent in sentences:
                if count_tokens(temp_chunk + " " + sent) <= max_tokens:
                    temp_chunk = (temp_chunk + " " + sent).strip()
                else:
                    if temp_chunk:
                        chunks.append(temp_chunk)
                        # Overlap: giữ lại phần cuối
                        words = temp_chunk.split()
                        overlap_text = " ".join(words[-overlap_tokens:]) if len(words) > overlap_tokens else temp_chunk
                        temp_chunk = overlap_text + " " + sent
                    else:
                        # Câu đơn quá dài → cắt cứng
                        temp_chunk = sent[:max_tokens * 4]  # rough char estimate
            
            if temp_chunk:
                chunks.append(temp_chunk.strip())
                
        elif current_tokens + section_tokens <= max_tokens:
            # Gộp section nhỏ vào chunk hiện tại
            current_chunk = (current_chunk + "\n\n" + section).strip()
        else:
            # Chunk hiện tại đã đủ → lưu và bắt đầu chunk mới
            chunks.append(current_chunk.strip())
            
            # Overlap: lấy phần cuối chunk trước
            words = current_chunk.split()
            if len(words) > overlap_tokens:
                overlap_text = " ".join(words[-overlap_tokens:])
                current_chunk = overlap_text + "\n\n" + section
            else:
                current_chunk = section
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def chunk_documents(documents: list[dict]) -> list[TextChunk]:
    """
    Chunking toàn bộ danh sách tài liệu.
    
    Args:
        documents: List[dict] với 'text', 'title', 'id'
        
    Returns:
        List[TextChunk] - Tất cả chunks từ tất cả tài liệu
    """
    all_chunks = []
    chunk_counter = 0
    
    for doc in documents:
        text = doc.get("text", "")
        doc_id = doc.get("id", 0)
        doc_title = doc.get("title", "")
        
        if not text:
            continue
        
        # Bước 1: Chia theo cấu trúc pháp lý
        sections = split_by_legal_structure(text)
        
        # Bước 2: Gộp/chia với overlap
        chunk_texts = chunk_with_overlap(sections)
        
        for ct in chunk_texts:
            # Tiêm metadata (Tên văn bản) trực tiếp vào nội dung chunk
            # Giúp bảo toàn ngữ cảnh, tránh việc LLM bị mất dấu nguồn gốc văn bản
            contextualized_text = f"[Văn bản: {doc_title}]\n{ct}"
            
            chunk = TextChunk(
                text=contextualized_text,
                doc_id=doc_id,
                chunk_id=chunk_counter,
                doc_title=doc_title,
                level=0,
                metadata={
                    "source": "UTS_VLC",
                    "tokens": count_tokens(contextualized_text),
                }
            )
            all_chunks.append(chunk)
            chunk_counter += 1
    
    console.print(
        f"[green]✂️ Đã tạo {len(all_chunks)} chunks từ {len(documents)} tài liệu[/]"
    )
    
    # Thống kê
    token_counts = [c.metadata["tokens"] for c in all_chunks]
    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        console.print(
            f"[dim]Tokens/chunk: min={min(token_counts)}, "
            f"max={max(token_counts)}, avg={avg_tokens:.0f}[/]"
        )
    
    return all_chunks


if __name__ == "__main__":
    # Test chunking
    sample_doc = {
        "id": 0,
        "title": "Luật Doanh nghiệp 2020",
        "text": """
Chương I: NHỮNG QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh
Luật này quy định về việc thành lập, tổ chức quản lý, tổ chức lại, giải thể và hoạt động có liên quan của doanh nghiệp, bao gồm công ty trách nhiệm hữu hạn, công ty cổ phần, công ty hợp danh và doanh nghiệp tư nhân; quy định về nhóm công ty.

Điều 2. Đối tượng áp dụng
1. Doanh nghiệp.
2. Cơ quan, tổ chức, cá nhân có liên quan đến việc thành lập, tổ chức quản lý, tổ chức lại, giải thể và hoạt động có liên quan của doanh nghiệp.

Điều 3. Áp dụng Luật Doanh nghiệp và các luật chuyên ngành
Trường hợp luật chuyên ngành có quy định đặc thù về việc thành lập, tổ chức quản lý, tổ chức lại, giải thể và hoạt động có liên quan của doanh nghiệp thì áp dụng quy định của luật đó.
"""
    }
    
    chunks = chunk_documents([sample_doc])
    for c in chunks:
        console.print(f"\n[bold]Chunk {c.chunk_id}[/] ({c.metadata['tokens']} tokens):")
        console.print(f"[dim]{c.text[:150]}...[/]")
