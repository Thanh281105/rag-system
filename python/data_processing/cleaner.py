"""
Làm sạch văn bản từ ArXiv papers (PDF extraction output).
Xử lý:
- Loại bỏ LaTeX artifacts
- Xóa headers/footers/page numbers
- Xóa References/Bibliography section
- Chuẩn hoá khoảng trắng
"""
import re
import unicodedata
from rich.console import Console

console = Console()


def normalize_unicode(text: str) -> str:
    """Chuẩn hoá Unicode NFC."""
    return unicodedata.normalize("NFC", text)


def remove_latex_artifacts(text: str) -> str:
    """Loại bỏ LaTeX commands và artifacts từ PDF extraction."""
    # Remove LaTeX commands: \textbf{...}, \textit{...}, etc.
    text = re.sub(r'\\text(?:bf|it|rm|sf|tt)\{([^}]*)\}', r'\1', text)
    # Remove \cite{...}, \ref{...}, \label{...}
    text = re.sub(r'\\(?:cite|ref|label|eqref|cref)\{[^}]*\}', '', text)
    # Remove \begin{...} and \end{...}
    text = re.sub(r'\\(?:begin|end)\{[^}]*\}', '', text)
    # Remove standalone backslash commands (but keep content)
    text = re.sub(r'\\[a-zA-Z]+\s*', ' ', text)
    # Remove $ math delimiters but keep content
    text = re.sub(r'\$([^$]*)\$', r'\1', text)
    # Remove double $$ math blocks
    text = re.sub(r'\$\$([^$]*)\$\$', r'\1', text)
    return text


def remove_pdf_noise(text: str) -> str:
    """Loại bỏ noise đặc trưng từ PDF extraction."""
    # Page numbers: standalone numbers on a line
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    # Headers/footers lặp lại (arXiv watermark)
    text = re.sub(r'arXiv:\d+\.\d+v?\d*\s*\[.*?\]\s*\d+\s*\w+\s*\d+', '', text)
    # Remove "Preprint." or "Under review." lines
    text = re.sub(r'^\s*(?:Preprint|Under review|Published|Submitted)\.?\s*$',
                  '', text, flags=re.MULTILINE | re.IGNORECASE)
    # Remove email addresses
    text = re.sub(r'[\w.-]+@[\w.-]+\.\w+', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove footnote markers
    text = re.sub(r'(?<!\d)\d{1,2}(?!\d)', '', text)
    return text


def remove_references_section(text: str) -> str:
    """
    Xóa phần References/Bibliography ở cuối paper.
    Phần này không cần thiết cho RAG — chỉ giữ nội dung chính.
    """
    # Tìm vị trí bắt đầu References
    patterns = [
        r'\n\s*References\s*\n',
        r'\n\s*REFERENCES\s*\n',
        r'\n\s*Bibliography\s*\n',
        r'\n\s*BIBLIOGRAPHY\s*\n',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            # Giữ content trước References
            text = text[:match.start()]
            break
    return text


def remove_appendix(text: str) -> str:
    """Xóa Appendix nếu có (thường ít giá trị cho RAG)."""
    patterns = [
        r'\n\s*(?:Appendix|APPENDIX|Supplementary Material)\s*\n',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            text = text[:match.start()]
            break
    return text


def clean_whitespace(text: str) -> str:
    """Chuẩn hoá khoảng trắng."""
    # Thay nhiều khoảng trắng liên tiếp thành 1
    text = re.sub(r'[ \t]+', ' ', text)
    # Thay nhiều dòng trống liên tiếp thành 1
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Fix hyphenation breaks (word- \n continuation)
    text = re.sub(r'-\s*\n\s*', '', text)
    return text.strip()


def clean_document(text: str) -> str:
    """
    Pipeline làm sạch hoàn chỉnh cho 1 văn bản ArXiv paper.
    """
    if not text or not isinstance(text, str):
        return ""

    text = normalize_unicode(text)
    text = remove_references_section(text)
    text = remove_appendix(text)
    text = remove_latex_artifacts(text)
    text = remove_pdf_noise(text)
    text = clean_whitespace(text)

    return text


def clean_documents(documents: list[dict]) -> list[dict]:
    """
    Làm sạch danh sách tài liệu ArXiv.

    Args:
        documents: List[dict] với key 'text' và 'title'

    Returns:
        List[dict] đã làm sạch
    """
    cleaned = []
    skipped = 0

    for doc in documents:
        clean_text = clean_document(doc.get("text", ""))
        clean_title = doc.get("title", "").strip()

        # Bỏ qua tài liệu quá ngắn (< 200 ký tự)
        if len(clean_text) < 200:
            skipped += 1
            continue

        cleaned.append({
            **doc,
            "text": clean_text,
            "title": clean_title,
        })

    console.print(f"[green]🧹 Đã làm sạch {len(cleaned)} papers, bỏ qua {skipped} papers quá ngắn[/]")
    return cleaned


if __name__ == "__main__":
    # Test với văn bản mẫu
    sample = """
    arXiv:2301.12345v2 [cs.AI] 15 Jan 2023

    Attention Is All You Need

    Ashish Vaswani, Noam Shazeer
    vaswani@google.com

    Abstract
    The dominant sequence transduction models are based on complex
    recurrent or convolutional neural networks.

    1 Introduction

    Recurrent neural networks, \\textbf{long short-term memory} and gated
    recurrent \\cite{cho2014learning} neural networks in particular.

    $\\alpha = softmax(QK^T / \\sqrt{d_k})V$

    References

    [1] Bahdanau et al. Neural machine translation. 2014.
    [2] Cho et al. Learning phrase representations. 2014.
    """
    result = clean_document(sample)
    console.print(f"[bold]Input:[/]\n{sample}")
    console.print(f"\n[bold]Output:[/]\n{result}")
