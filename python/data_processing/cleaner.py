"""
Làm sạch văn bản pháp lý Việt Nam.
Xử lý toàn diện các vấn đề:
- Chuẩn hoá Unicode (NFC cho tiếng Việt)
- Loại bỏ header/footer nhà nước lặp lại
- Loại bỏ artifacts OCR (số trang, watermark)
- Xử lý chữ ký/nơi nhận/preamble
- Chuẩn hoá khoảng trắng
"""
import re
import unicodedata
from rich.console import Console

console = Console()


def normalize_unicode(text: str) -> str:
    """Chuẩn hoá Unicode NFC cho tiếng Việt."""
    return unicodedata.normalize("NFC", text)


def remove_special_chars(text: str) -> str:
    """Loại bỏ ký tự đặc biệt không cần thiết nhưng giữ dấu câu pháp lý."""
    # Giữ lại: chữ cái, số, dấu câu cơ bản, dấu tiếng Việt, dấu đặc thù pháp lý
    text = re.sub(r'[^\w\s.,;:!?()/"\'\-–—§đĐ%°₫\[\]]', ' ', text)
    return text


def clean_whitespace(text: str) -> str:
    """Chuẩn hoá khoảng trắng: bỏ thừa, giữ xuống dòng có ý nghĩa."""
    # Thay nhiều khoảng trắng liên tiếp thành 1
    text = re.sub(r'[ \t]+', ' ', text)
    # Thay nhiều dòng trống liên tiếp thành 1
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


def clean_legal_artifacts(text: str) -> str:
    """Loại bỏ artifacts đặc trưng của văn bản luật VN (scan/OCR/PDF)."""
    
    # ─── Header nhà nước ─────────────────────────────────────
    # "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM / Độc lập - Tự do - Hạnh phúc"
    text = re.sub(
        r'CỘNG\s+HÒA\s+XÃ\s+HỘI\s+CHỦ\s+NGHĨA\s+VIỆT\s+NAM\s*'
        r'(?:Độc\s+lập\s*-\s*Tự\s+do\s*-\s*Hạnh\s+phúc)?\s*[-─—]+',
        '', text, flags=re.IGNORECASE
    )
    
    # ─── Số trang ────────────────────────────────────────────
    text = re.sub(r'Trang\s+\d+\s*/\s*\d+', '', text)
    text = re.sub(r'-\s*\d+\s*-', '', text)
    text = re.sub(r'\[\s*\d+\s*\]', '', text)  # [1], [2] footnote markers
    
    # ─── Watermark OCR ───────────────────────────────────────
    text = re.sub(r'(?:Bản\s+sao|Không\s+chỉnh\s+sửa|Lưu\s+hành\s+nội\s+bộ)', 
                  '', text, flags=re.IGNORECASE)
    
    # ─── Chữ ký / Nơi nhận (cuối văn bản) ────────────────────
    # Loại bỏ block "Nơi nhận:" và nội dung sau nó
    text = re.sub(
        r'\n\s*Nơi\s+nhận\s*:.*',
        '', text, flags=re.DOTALL | re.IGNORECASE
    )
    
    # Loại bỏ dòng ký tên quan chức (TM., KT., Q., PHÓ, TỔNG, BỘ TRƯỞNG...)
    text = re.sub(
        r'\n\s*(?:TM\.|KT\.|Q\.)\s*(?:CHÍNH PHỦ|QUỐC HỘI|THỦ TƯỚNG|'
        r'BỘ TRƯỞNG|CHỦ TỊCH|TỔNG GIÁM ĐỐC).*',
        '', text, flags=re.DOTALL | re.IGNORECASE
    )
    
    # ─── Footnotes / Chú thích ───────────────────────────────
    # Pattern "(*)" hoặc "(1)" hoặc số superscript ở cuối dòng
    text = re.sub(r'\(\*+\)', '', text)
    
    # ─── Metadata header (số hiệu văn bản ở đầu) ────────────
    # "Số: 59/2020/QH14" → giữ lại nhưng chuẩn hoá
    # Không xoá vì có thể hữu ích cho retrieval
    
    # ─── Dòng "Căn cứ..." lặp lại (preamble) ────────────────
    # Không xoá hoàn toàn, chỉ gom nhóm các dòng "Căn cứ" thành 1 đoạn rõ ràng
    # (để chunker có thể tách riêng phần preamble)
    
    return text


def normalize_legal_numbering(text: str) -> str:
    """
    Chuẩn hoá numbering pháp lý để chunker nhận diện chính xác hơn.
    Ví dụ: "Điều1." → "Điều 1.", "khoản1" → "khoản 1"
    """
    # Chuẩn hoá "Điều" (đảm bảo có space sau)
    text = re.sub(r'(Điều)\s*(\d+)', r'\1 \2', text)
    
    # Chuẩn hoá "Chương" 
    text = re.sub(r'(Chương)\s*([IVXLCDM]+)', r'\1 \2', text)
    
    # Chuẩn hoá "Mục"
    text = re.sub(r'(Mục)\s*(\d+)', r'\1 \2', text)
    
    # Chuẩn hoá Khoản: "1." phải có space trước số ở đầu dòng
    # (không thay đổi "1." ở giữa câu - chỉ khi ở đầu dòng)
    
    return text


def clean_document(text: str) -> str:
    """
    Pipeline làm sạch hoàn chỉnh cho 1 văn bản pháp lý.
    
    Args:
        text: Văn bản gốc
        
    Returns:
        Văn bản đã được làm sạch
    """
    if not text or not isinstance(text, str):
        return ""
    
    text = normalize_unicode(text)
    text = clean_legal_artifacts(text)
    text = normalize_legal_numbering(text)
    text = remove_special_chars(text)
    text = clean_whitespace(text)
    
    return text


def clean_documents(documents: list[dict]) -> list[dict]:
    """
    Làm sạch danh sách tài liệu.
    
    Args:
        documents: List[dict] với key 'text' và 'title'
        
    Returns:
        List[dict] đã làm sạch
    """
    cleaned = []
    skipped = 0
    
    for doc in documents:
        clean_text = clean_document(doc.get("text", ""))
        clean_title = clean_document(doc.get("title", ""))
        
        # Bỏ qua tài liệu quá ngắn (< 50 ký tự)
        if len(clean_text) < 50:
            skipped += 1
            continue
        
        cleaned.append({
            **doc,
            "text": clean_text,
            "title": clean_title,
        })
    
    console.print(f"[green]🧹 Đã làm sạch {len(cleaned)} tài liệu, bỏ qua {skipped} tài liệu quá ngắn[/]")
    return cleaned


if __name__ == "__main__":
    # Test với văn bản mẫu
    sample = """
    CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
    Độc lập - Tự do - Hạnh phúc
    --------
    
    Số: 59/2020/QH14
    
    Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam;
    Căn cứ Luật Ban hành văn bản quy phạm pháp luật;
    
    Điều1.   Phạm vi   điều chỉnh
    
    Luật này quy định    về việc thành lập,  tổ chức quản lý, 
    tổ chức lại,  giải thể  và hoạt động có liên quan 
    của doanh    nghiệp.   Trang 1/25
    
    Bản sao
    
    Nơi nhận:
    - Thủ tướng Chính phủ;
    - Các bộ, cơ quan ngang bộ;
    
    TM. CHÍNH PHỦ
    THỦ TƯỚNG
    Nguyễn Xuân Phúc
    """
    result = clean_document(sample)
    console.print(f"[bold]Input:[/]\n{sample}")
    console.print(f"\n[bold]Output:[/]\n{result}")
