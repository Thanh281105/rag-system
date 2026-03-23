"""
Phân mảnh văn bản pháp lý thành các đoạn (chunks) phù hợp cho embedding.
Hỗ trợ chunking theo cấu trúc phân cấp đầy đủ của luật Việt Nam:
  Phần → Chương → Mục → Tiểu mục → Điều → Khoản → Điểm

Tính năng:
- Nhận diện đầy đủ cấu trúc: Phần, Chương, Mục, Tiểu mục, Điều, Khoản, Điểm
- Parent Context Injection: tự động ghép tiêu đề Điều cha vào chunk con
- Token-based overlap (dùng tiktoken, không dùng word-split)
- Context injection: ghép tên văn bản vào mỗi chunk
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


def get_overlap_text(text: str, overlap_tokens: int) -> str:
    """
    Lấy phần cuối của text theo đúng số tokens (không dùng word-split).
    Đảm bảo overlap chính xác theo token count.
    """
    tokens = _ENCODER.encode(text)
    if len(tokens) <= overlap_tokens:
        return text
    overlap_token_ids = tokens[-overlap_tokens:]
    return _ENCODER.decode(overlap_token_ids)


# ─── Regex patterns cho cấu trúc pháp lý VN ─────────────────────────
# Từ đơn vị lớn đến nhỏ
PATTERN_PHAN = re.compile(
    r'^\s*(?:PHẦN\s+THỨ\s+(?:NHẤT|HAI|BA|BỐN|NĂM|SÁU|BẢY|TÁM|CHÍN|MƯỜI'
    r'|MƯỜI\s+MỘT|MƯỜI\s+HAI|MƯỜI\s+BA|MƯỜI\s+BỐN|MƯỜI\s+LĂM)|PHẦN\s+[IVXLCDM]+)',
    re.MULTILINE | re.IGNORECASE
)
PATTERN_CHUONG = re.compile(
    r'^\s*(?:CHƯƠNG|Chương)\s+[IVXLCDM]+',
    re.MULTILINE
)
PATTERN_MUC = re.compile(
    r'^\s*(?:MỤC|Mục)\s+\d+',
    re.MULTILINE
)
PATTERN_TIEU_MUC = re.compile(
    r'^\s*(?:TIỂU MỤC|Tiểu mục)\s+\d+',
    re.MULTILINE
)
PATTERN_DIEU = re.compile(
    r'^\s*Điều\s+\d+[a-z]?\.?\s',
    re.MULTILINE
)
PATTERN_KHOAN = re.compile(
    r'^\s*\d+\.\s',
    re.MULTILINE
)
PATTERN_DIEM = re.compile(
    r'^\s*[a-zđ]\)\s',
    re.MULTILINE
)


def extract_dieu_title(text: str) -> str:
    """
    Trích xuất tiêu đề Điều từ một đoạn văn bản.
    Ví dụ: "Điều 134. Tội cố ý gây thương tích hoặc gây tổn hại cho sức khỏe"
    → trả về "Điều 134. Tội cố ý gây thương tích hoặc gây tổn hại cho sức khỏe"
    """
    match = re.match(r'^\s*(Điều\s+\d+[a-z]?\.?\s*[^\n]*)', text.strip())
    if match:
        title = match.group(1).strip()
        # Lấy dòng đầu tiên (tiêu đề Điều thường nằm trên 1 dòng)
        title = title.split('\n')[0].strip()
        # Cắt bớt nếu quá dài (>100 ký tự)
        if len(title) > 100:
            title = title[:100] + "..."
        return title
    return ""


def split_by_legal_structure(text: str) -> list[dict]:
    """
    Chia văn bản theo cấu trúc pháp lý phân cấp đầy đủ.
    
    Trả về list[dict] với keys:
        - 'text': nội dung đoạn
        - 'parent_dieu': tiêu đề Điều cha (nếu chunk ở cấp Khoản/Điểm)
    
    Chiến lược phân chia (ưu tiên giữ nguyên cấu trúc logic):
    1. Chia theo Điều trước
    2. Nếu một Điều quá dài (> CHUNK_SIZE tokens): chia tiếp theo Khoản
    3. Nếu một Khoản vẫn quá dài: chia tiếp theo Điểm
    4. Nếu vẫn quá dài: fallback chia theo câu
    """
    # ─── Bước 1: Chia theo Điều ──────────────────────────────
    dieu_parts = re.split(r'(?=\n\s*Điều\s+\d+[a-z]?\.?\s)', text)
    dieu_parts = [p.strip() for p in dieu_parts if p.strip()]
    
    # Nếu không tìm thấy cấu trúc Điều → thử chia theo Chương/Mục
    if len(dieu_parts) <= 1:
        for pattern in [PATTERN_CHUONG, PATTERN_MUC]:
            parts = re.split(f'(?={pattern.pattern})', text, flags=pattern.flags)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) > 1:
                return [{"text": p, "parent_dieu": ""} for p in parts]
        
        # Không tìm thấy cấu trúc nào → chia theo đoạn văn
        parts = text.split('\n\n')
        return [{"text": p.strip(), "parent_dieu": ""} 
                for p in parts if p.strip()]
    
    # ─── Bước 2: Kiểm tra từng Điều, chia tiếp nếu quá dài ─────
    result = []
    
    for dieu_text in dieu_parts:
        dieu_tokens = count_tokens(dieu_text)
        
        if dieu_tokens <= CHUNK_SIZE:
            # Điều ngắn gọn → giữ nguyên
            result.append({"text": dieu_text, "parent_dieu": ""})
            continue
        
        # Điều quá dài → chia theo Khoản
        dieu_title = extract_dieu_title(dieu_text)
        khoan_parts = re.split(r'(?=\n\s*\d+\.\s)', dieu_text)
        khoan_parts = [p.strip() for p in khoan_parts if p.strip()]
        
        if len(khoan_parts) <= 1:
            # Không có cấu trúc Khoản → giữ nguyên (sẽ bị chia ở bước overlap)
            result.append({"text": dieu_text, "parent_dieu": ""})
            continue
        
        for khoan_text in khoan_parts:
            khoan_tokens = count_tokens(khoan_text)
            
            if khoan_tokens <= CHUNK_SIZE:
                # Khoản vừa đủ → inject parent context
                result.append({
                    "text": khoan_text,
                    "parent_dieu": dieu_title,
                })
                continue
            
            # Khoản quá dài → chia theo Điểm
            diem_parts = re.split(r'(?=\n\s*[a-zđ]\)\s)', khoan_text)
            diem_parts = [p.strip() for p in diem_parts if p.strip()]
            
            if len(diem_parts) > 1:
                for diem_text in diem_parts:
                    result.append({
                        "text": diem_text,
                        "parent_dieu": dieu_title,
                    })
            else:
                # Không có Điểm → giữ nguyên, sẽ chia ở bước overlap
                result.append({
                    "text": khoan_text,
                    "parent_dieu": dieu_title,
                })
    
    return result


def chunk_with_overlap(
    sections: list[dict],
    max_tokens: int = CHUNK_SIZE,
    overlap_tokens: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Gộp các section nhỏ hoặc chia section lớn, đảm bảo token-based overlap.
    
    Args:
        sections: Danh sách dict với 'text' và 'parent_dieu'
        max_tokens: Số tokens tối đa mỗi chunk
        overlap_tokens: Số tokens overlap giữa các chunk
    
    Returns:
        Danh sách dict chunks đã xử lý
    """
    chunks = []
    current_chunk = ""
    current_parent = ""
    
    for section in sections:
        sec_text = section["text"]
        sec_parent = section.get("parent_dieu", "")
        section_tokens = count_tokens(sec_text)
        current_tokens = count_tokens(current_chunk)
        
        if section_tokens > max_tokens:
            # Section quá dài → chia nhỏ theo câu
            if current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "parent_dieu": current_parent
                })
                current_chunk = ""
            
            # Chia theo câu (regex phù hợp tiếng Việt: dấu chấm, chấm phẩy)
            sentences = re.split(r'(?<=[.!?;])\s+', sec_text)
            temp_chunk = ""
            
            for sent in sentences:
                if count_tokens(temp_chunk + " " + sent) <= max_tokens:
                    temp_chunk = (temp_chunk + " " + sent).strip()
                else:
                    if temp_chunk:
                        chunks.append({
                            "text": temp_chunk,
                            "parent_dieu": sec_parent
                        })
                        # Token-based overlap
                        temp_chunk = get_overlap_text(temp_chunk, overlap_tokens) + " " + sent
                    else:
                        # Câu đơn quá dài → cắt theo tokens
                        tokens = _ENCODER.encode(sent)
                        temp_chunk = _ENCODER.decode(tokens[:max_tokens])
            
            if temp_chunk:
                chunks.append({
                    "text": temp_chunk.strip(),
                    "parent_dieu": sec_parent
                })
            
            current_parent = sec_parent
                
        elif current_tokens + section_tokens <= max_tokens:
            # Gộp section nhỏ vào chunk hiện tại
            current_chunk = (current_chunk + "\n\n" + sec_text).strip()
            # Giữ parent_dieu từ section đầu tiên có parent
            if sec_parent and not current_parent:
                current_parent = sec_parent
        else:
            # Chunk hiện tại đã đủ → lưu và bắt đầu chunk mới
            chunks.append({
                "text": current_chunk.strip(),
                "parent_dieu": current_parent
            })
            
            # Token-based overlap: lấy phần cuối chunk trước
            overlap_text = get_overlap_text(current_chunk, overlap_tokens)
            current_chunk = overlap_text + "\n\n" + sec_text
            current_parent = sec_parent
    
    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "parent_dieu": current_parent
        })
    
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
        
        # Bước 1: Chia theo cấu trúc pháp lý phân cấp
        sections = split_by_legal_structure(text)
        
        # Bước 2: Gộp/chia với token-based overlap
        chunk_dicts = chunk_with_overlap(sections)
        
        for cd in chunk_dicts:
            ct = cd["text"]
            parent_dieu = cd.get("parent_dieu", "")
            
            # Context Injection: ghép tên văn bản + Điều cha vào chunk
            context_parts = [f"[Văn bản: {doc_title}]"]
            if parent_dieu:
                context_parts.append(f"[{parent_dieu}]")
            context_prefix = " ".join(context_parts)
            contextualized_text = f"{context_prefix}\n{ct}"
            
            chunk = TextChunk(
                text=contextualized_text,
                doc_id=doc_id,
                chunk_id=chunk_counter,
                doc_title=doc_title,
                level=0,
                metadata={
                    "source": "UTS_VLC",
                    "tokens": count_tokens(contextualized_text),
                    "parent_dieu": parent_dieu,
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
    
    # Thống kê parent context injection
    with_parent = sum(1 for c in all_chunks if c.metadata.get("parent_dieu"))
    if with_parent:
        console.print(
            f"[dim]Parent context injected: {with_parent}/{len(all_chunks)} chunks[/]"
        )
    
    return all_chunks


if __name__ == "__main__":
    # Test chunking với văn bản có cấu trúc Khoản/Điểm
    sample_doc = {
        "id": 0,
        "title": "Bộ luật Hình sự 2015",
        "text": """
Chương XIV: CÁC TỘI XÂM PHẠM SỞ HỮU

Điều 134. Tội cố ý gây thương tích hoặc gây tổn hại cho sức khỏe của người khác
1. Người nào cố ý gây thương tích hoặc gây tổn hại cho sức khỏe của người khác mà tỷ lệ tổn thương cơ thể từ 11% đến 30% hoặc dưới 11% nhưng thuộc một trong các trường hợp sau đây, thì bị phạt cải tạo không giam giữ đến 03 năm hoặc phạt tù từ 06 tháng đến 03 năm:
a) Dùng vũ khí, vật liệu nổ, hung khí nguy hiểm hoặc thủ đoạn có khả năng gây nguy hại cho nhiều người;
b) Dùng axit nguy hiểm hoặc hoá chất nguy hiểm;
c) Đối với người dưới 16 tuổi, phụ nữ mà biết là có thai;
d) Đối với người đang thi hành công vụ.

2. Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 02 năm đến 06 năm:
a) Gây thương tích hoặc gây tổn hại cho sức khỏe của người khác mà tỷ lệ tổn thương cơ thể từ 31% đến 60%;
b) Phạm tội đối với 02 người trở lên mà tỷ lệ tổn thương cơ thể của mỗi người từ 11% đến 30%.

Điều 135. Tội cố ý gây thương tích hoặc gây tổn hại cho sức khỏe của người khác trong trạng thái tinh thần bị kích động mạnh
1. Người nào cố ý gây thương tích trong trạng thái tinh thần bị kích động mạnh do hành vi trái pháp luật nghiêm trọng của nạn nhân, thì bị phạt tiền từ 10.000.000 đồng đến 50.000.000 đồng.
"""
    }
    
    chunks = chunk_documents([sample_doc])
    for c in chunks:
        console.print(f"\n[bold]Chunk {c.chunk_id}[/] ({c.metadata['tokens']} tokens):")
        parent = c.metadata.get('parent_dieu', '')
        if parent:
            console.print(f"[yellow]  Parent: {parent}[/]")
        console.print(f"[dim]{c.text[:200]}...[/]")
