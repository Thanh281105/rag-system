//! Router Agent - Phân loại ý định câu hỏi
//! Quyết định: cần tra cứu luật (legal) hay chỉ giao tiếp thường (casual)

use anyhow::Result;
use tracing::info;

use crate::services::groq::{GroqService, ModelTier};

#[derive(Debug, Clone, PartialEq)]
pub enum QueryIntent {
    /// Câu hỏi pháp lý → cần RAG pipeline
    Legal,
    /// Giao tiếp thông thường → trả lời trực tiếp
    Casual,
}

const ROUTER_SYSTEM_PROMPT: &str = r#"
Bạn là Router Agent trong hệ thống trả lời câu hỏi pháp lý Việt Nam.
Nhiệm vụ: phân loại ý định câu hỏi.

Trả lời ĐÚNG MỘT từ:
- "LEGAL" nếu câu hỏi liên quan đến luật, nghị định, thông tư, điều khoản, quy định pháp lý
- "CASUAL" nếu là giao tiếp thông thường, chào hỏi, hoặc không liên quan pháp lý

Chỉ trả lời LEGAL hoặc CASUAL, không giải thích.
"#;

pub struct RouterAgent;

impl RouterAgent {
    /// Phân loại ý định câu hỏi
    pub async fn classify(groq: &GroqService, question: &str) -> Result<QueryIntent> {
        let response = groq
            .chat(
                ModelTier::Fast,
                ROUTER_SYSTEM_PROMPT,
                question,
                0.0,
                10,
            )
            .await?;

        let intent = if response.trim().to_uppercase().contains("LEGAL") {
            QueryIntent::Legal
        } else {
            QueryIntent::Casual
        };

        info!("Router: {:?} for '{}'", intent, &question[..question.floor_char_boundary(50)]);
        Ok(intent)
    }

    /// Trả lời casual (không cần RAG)
    pub async fn casual_response(groq: &GroqService, question: &str) -> Result<String> {
        groq.chat(
            ModelTier::Smart,
            "Bạn là trợ lý pháp lý thân thiện. Trả lời ngắn gọn, lịch sự bằng tiếng Việt. \
             Nếu người dùng hỏi gì liên quan pháp lý, khuyên họ hỏi lại cụ thể hơn.",
            question,
            0.7,
            256,
        )
        .await
    }
}
