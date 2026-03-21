"""
Làm sạch văn bản pháp lý Việt Nam.
Xử lý các vấn đề phổ biến: ký tự đặc biệt, Unicode, khoảng trắng thừa.
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
    # Giữ lại: chữ cái, số, dấu câu cơ bản, dấu tiếng Việt
    text = re.sub(r'[^\w\s.,;:!?()/"\'–\-§đĐ]', ' ', text)
    return text


def clean_whitespace(text: str) -> str:
    """Chuẩn hoá khoảng trắng: bỏ thừa, giữ xuống dòng có ý nghĩa."""
    # Thay nhiều khoảng trắng liên tiếp thành 1
    text = re.sub(r'[ \t]+', ' ', text)
    # Thay nhiều dòng trống liên tiếp thành 1
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


def clean_legal_artifacts(text: str) -> str:
    """Loại bỏ artifacts đặc trưng của văn bản luật scan/OCR."""
    # Bỏ số trang
    text = re.sub(r'Trang \d+/\d+', '', text)
    text = re.sub(r'- \d+ -', '', text)
    # Bỏ header/footer lặp lại
    text = re.sub(r'(CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\s*Độc lập - Tự do - Hạnh phúc\s*-+)', '', text)
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
    
    Điều 1.   Phạm vi   điều chỉnh
    
    Luật này quy định    về việc thành lập,  tổ chức quản lý, 
    tổ chức lại,  giải thể  và hoạt động có liên quan 
    của doanh    nghiệp.   Trang 1/25
    """
    result = clean_document(sample)
    console.print(f"[bold]Input:[/]\n{sample}")
    console.print(f"\n[bold]Output:[/]\n{result}")
