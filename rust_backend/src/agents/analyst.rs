//! Analyst Agent (Agent 2) - Sinh câu trả lời + Self-check
//! Input: Top-K context (EN) + Original query (VN)
//! Output: Answer bằng tiếng Việt với citations

use anyhow::Result;
use tracing::info;

use crate::models::document::ArxivDocument;
use crate::services::groq::{GroqService, ModelTier};

const ANALYST_SYSTEM_PROMPT: &str = r#"
Bạn là trợ lý nghiên cứu AI chuyên nghiệp.
Nhiệm vụ: Trả lời câu hỏi tiếng Việt dựa TRỰC TIẾP và DUY NHẤT trên các "Evidence" (bằng chứng tiếng Anh) được cung cấp.

QUY TẮC BẮT BUỘC:
1. Chỉ sử dụng thông tin có trong Evidence. KHÔNG ĐƯỢC bịa đặt hay suy diễn ngoài evidence.
2. Giữ nguyên thuật ngữ kỹ thuật bằng tiếng Anh (Transformer, attention, LoRA, fine-tuning, etc.)
3. Trích dẫn theo format: Theo "[Tên Paper]" (Author, Year), ...
4. Nếu Evidence không đủ để trả lời → nói rõ "Tôi không tìm thấy thông tin này trong cơ sở dữ liệu"
5. Câu trả lời phải bằng TIẾNG VIỆT, rõ ràng và có cấu trúc

YÊU CẦU VỀ ĐỘ CHI TIẾT (BẮT BUỘC):
- PHẢI trích dẫn SỐ LIỆU CỤ THỂ nếu Evidence có (accuracy, F1, BLEU, parameters, latency, etc.)
- PHẢI mô tả PHƯƠNG PHÁP (methodology) mà paper sử dụng, không chỉ nêu kết luận
- PHẢI giải thích TẠI SAO (reasoning) đằng sau kết quả, nếu Evidence có đề cập
- KHÔNG ĐƯỢC chỉ paraphrase tiêu đề paper rồi kết luận chung chung
- Nếu Evidence chứa bảng so sánh, thí nghiệm, hoặc ablation study → tóm tắt kết quả chính

CẤU TRÚC CÂU TRẢ LỜI:
1. Tóm tắt ngắn (1-2 câu trả lời trực tiếp)
2. Chi tiết phương pháp và kết quả (với số liệu cụ thể từ Evidence)
3. Kết luận/Hạn chế nếu Evidence có đề cập

SELF-CHECK (tự kiểm tra trước khi trả lời):
- Các con số (accuracy, F1, parameters) có khớp CHÍNH XÁC với Evidence không?
- Tên model/method có đúng không?
- Kết luận có được support bởi Evidence không?
- Câu trả lời có đủ chi tiết hay chỉ đang paraphrase tiêu đề?
"#;

pub struct AnalystAgent;

impl AnalystAgent {
    /// Sinh câu trả lời phân tích từ evidence ArXiv
    pub async fn analyze(
        groq: &GroqService,
        question_vn: &str,
        evidence: &[ArxivDocument],
    ) -> Result<String> {
        if evidence.is_empty() {
            return Ok(
                "Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu. \
                 Vui lòng thử hỏi lại với câu hỏi cụ thể hơn hoặc về chủ đề AI/ML khác."
                    .to_string(),
            );
        }

        // Tạo evidence text từ ArXiv documents
        let evidence_text: String = evidence
            .iter()
            .enumerate()
            .map(|(i, doc)| {
                format!(
                    "[Evidence {}] ({} by {}, {}, arXiv:{}, Score: {:.2}):\n{}",
                    i + 1,
                    doc.doc_title,
                    doc.authors,
                    doc.year,
                    doc.arxiv_id,
                    doc.score,
                    &doc.text[..doc.text.floor_char_boundary(2000)],
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");

        let user_prompt = format!(
            "CÂU HỎI (tiếng Việt): {question_vn}\n\n\
             EVIDENCE (tiếng Anh — từ ArXiv papers):\n{evidence_text}\n\n\
             Hãy phân tích và trả lời câu hỏi bằng TIẾNG VIỆT dựa trên evidence trên. \
             Nhớ trích dẫn theo format: Theo [Paper Title] (Author, Year), ..."
        );

        let answer = groq
            .chat(ModelTier::Smart, ANALYST_SYSTEM_PROMPT, &user_prompt, 0.1, 2048)
            .await?;

        info!("Analyst: generated {} chars answer", answer.len());
        Ok(answer)
    }

    /// Kiểm tra xem query có cần Reviewer không (heuristic)
    pub fn needs_review(question: &str) -> bool {
        let question_lower = question.to_lowercase();
        let trigger_keywords = [
            // Numbers / metrics
            "accuracy", "f1", "precision", "recall", "bleu", "rouge",
            "bao nhiêu", "số liệu", "kết quả", "hiệu suất", "tỷ lệ",
            "parameter", "params", "flops", "latency",
            // Formulas
            "công thức", "formula", "equation",
            // Cost / performance comparison
            "so sánh", "compare", "tốt hơn", "better", "worse",
            "chi phí", "cost", "giá",
        ];

        trigger_keywords.iter().any(|kw| question_lower.contains(kw))
    }
}
