//! Rust text processing extension cho Python (via PyO3).
//! Tăng tốc cleaning và chunking văn bản pháp lý.

use pyo3::prelude::*;
use regex::Regex;
use unicode_normalization::UnicodeNormalization;

/// Chuẩn hoá Unicode NFC cho tiếng Việt
#[pyfunction]
fn normalize_unicode(text: &str) -> String {
    text.nfc().collect::<String>()
}

/// Loại bỏ ký tự đặc biệt, giữ dấu câu pháp lý
#[pyfunction]
fn remove_special_chars(text: &str) -> String {
    let re = Regex::new(r#"[^\w\s.,;:!?()/\"'\-–§]"#).unwrap();
    re.replace_all(text, " ").to_string()
}

/// Chuẩn hoá khoảng trắng
#[pyfunction]
fn clean_whitespace(text: &str) -> String {
    let re_spaces = Regex::new(r"[ \t]+").unwrap();
    let re_lines = Regex::new(r"\n\s*\n").unwrap();

    let result = re_spaces.replace_all(text, " ");
    let result = re_lines.replace_all(&result, "\n\n");
    result.trim().to_string()
}

/// Loại bỏ artifacts pháp lý (header/footer, số trang)
#[pyfunction]
fn clean_legal_artifacts(text: &str) -> String {
    let re_page = Regex::new(r"Trang \d+/\d+").unwrap();
    let re_page2 = Regex::new(r"- \d+ -").unwrap();
    let re_header = Regex::new(
        r"CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\s*Độc lập - Tự do - Hạnh phúc\s*-+",
    )
    .unwrap();

    let result = re_page.replace_all(text, "");
    let result = re_page2.replace_all(&result, "");
    let result = re_header.replace_all(&result, "");
    result.to_string()
}

/// Pipeline làm sạch hoàn chỉnh (gọi tất cả hàm trên)
#[pyfunction]
fn clean_text(text: &str) -> String {
    let result = normalize_unicode(text);
    let result = clean_legal_artifacts(&result);
    let result = remove_special_chars(&result);
    clean_whitespace(&result)
}

/// Chia văn bản theo Điều/Chương/Mục
#[pyfunction]
fn split_by_legal_structure(text: &str) -> Vec<String> {
    // Thử chia theo Điều
    let re_dieu = Regex::new(r"(?m)(?=\n\s*Điều\s+\d+)").unwrap();
    let parts: Vec<&str> = re_dieu.split(text).collect();

    if parts.len() > 1 {
        return parts.iter().map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect();
    }

    // Thử chia theo Chương
    let re_chuong = Regex::new(r"(?m)(?=\n\s*Chương\s+[IVXLCDM]+)").unwrap();
    let parts: Vec<&str> = re_chuong.split(text).collect();

    if parts.len() > 1 {
        return parts.iter().map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect();
    }

    // Fallback: chia theo đoạn
    text.split("\n\n")
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

/// Batch clean nhiều văn bản (tận dụng song song)
#[pyfunction]
fn batch_clean(texts: Vec<String>) -> Vec<String> {
    texts.iter().map(|t| clean_text(t)).collect()
}

/// Module PyO3
#[pymodule]
fn rust_text_processor(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(normalize_unicode, m)?)?;
    m.add_function(wrap_pyfunction!(remove_special_chars, m)?)?;
    m.add_function(wrap_pyfunction!(clean_whitespace, m)?)?;
    m.add_function(wrap_pyfunction!(clean_legal_artifacts, m)?)?;
    m.add_function(wrap_pyfunction!(clean_text, m)?)?;
    m.add_function(wrap_pyfunction!(split_by_legal_structure, m)?)?;
    m.add_function(wrap_pyfunction!(batch_clean, m)?)?;
    Ok(())
}
