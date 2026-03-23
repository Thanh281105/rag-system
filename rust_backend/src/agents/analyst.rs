//! Analyst Agent - Sinh câu trả lời lập luận từng bước
//! Đọc bằng chứng từ RAG Agent và tạo câu trả lời có cấu trúc

use anyhow::Result;
use tracing::info;

use crate::models::document::LegalDocument;
use crate::services::groq::{GroqService, ModelTier};

const ANALYST_SYSTEM_PROMPT: &str = r#"
Bạn là Analyst Agent - chuyên gia phân tích pháp luật Việt Nam.

NHIỆM VỤ: Dựa trên các bằng chứng pháp lý được cung cấp, hãy trả lời câu hỏi 
một cách chính xác, có cấu trúc, và lập luận từng bước.

QUY TẮC:
1. CHỈ trả lời dựa trên thông tin ĐƯỢC CUNG CẤP TRONG BẰNG CHỨNG. TUYỆT ĐỐI KHÔNG dùng kiến thức bên ngoài.
2. CHỐNG ẢO GIÁC ĐIỀU LUẬT: Không được gán một con số ngẫu nhiên trong phần text thành "Điều X" hoặc "Khoản Y" nếu văn bản không ghi chính xác cụm từ đó. 
3. Nếu không chắc chắn về số Điều/Khoản, hãy trích dẫn nguyên văn tiêu đề hoặc đoạn văn đó.
4. Nếu bằng chứng không đủ, nói rõ "Dựa trên thông tin hiện có..."
5. Lập luận theo cấu trúc: Cơ sở pháp lý → Phân tích → Kết luận
6. Viết bằng tiếng Việt, văn phong chuyên nghiệp.
"#;

pub struct AnalystAgent;

impl AnalystAgent {
    /// Sinh câu trả lời phân tích từ bằng chứng pháp lý
    pub async fn analyze(
        groq: &GroqService,
        question: &str,
        evidence: &[LegalDocument],
    ) -> Result<String> {
        // Tạo context từ evidence
        let evidence_text: String = evidence
            .iter()
            .enumerate()
            .map(|(i, doc)| {
                format!(
                    "[Bằng chứng {}] (Nguồn: {}, Score: {:.2}):\n{}",
                    i + 1,
                    doc.doc_title,
                    doc.score,
                    &doc.text[..doc.text.floor_char_boundary(800)],
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");

        let user_prompt = format!(
            "CÂU HỎI: {question}\n\n\
             BẰNG CHỨNG PHÁP LÝ:\n{evidence_text}\n\n\
             Hãy phân tích và trả lời câu hỏi dựa trên bằng chứng trên."
        );

        let answer = groq
            .chat(ModelTier::Smart, ANALYST_SYSTEM_PROMPT, &user_prompt, 0.1, 2048)
            .await?;

        info!("Analyst: generated {} chars answer", answer.len());
        Ok(answer)
    }
}
