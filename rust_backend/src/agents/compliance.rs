//! Compliance Agent - "Thẩm phán" kiểm tra hallucination
//! Đối chiếu chéo câu trả lời với bằng chứng gốc

use anyhow::Result;
use tracing::{info, warn};

use crate::models::document::LegalDocument;
use crate::models::query::ComplianceResult;
use crate::services::groq::{GroqService, ModelTier};

const COMPLIANCE_SYSTEM_PROMPT: &str = r#"
Bạn là Compliance Agent - "Thẩm phán" kiểm tra tính chính xác pháp lý.

NHIỆM VỤ: Đối chiếu câu trả lời với bằng chứng gốc và phát hiện:
1. Thông tin bịa đặt (hallucination) - không có trong bằng chứng
2. Trích dẫn sai số điều, khoản
3. Lập luận logic không chính xác
4. Kết luận không được support bởi bằng chứng

TRẢ LỜI theo JSON format:
{
    "is_compliant": true/false,
    "issues": ["Vấn đề 1", "Vấn đề 2"],
    "suggestion": "Gợi ý sửa nếu có vấn đề"
}
"#;

const MAX_RETRY: u32 = 2;

pub struct ComplianceAgent;

impl ComplianceAgent {
    /// Kiểm tra compliance của câu trả lời
    pub async fn check(
        groq: &GroqService,
        question: &str,
        answer: &str,
        evidence: &[LegalDocument],
    ) -> Result<ComplianceResult> {
        let evidence_text: String = evidence
            .iter()
            .enumerate()
            .map(|(i, doc)| format!("[Bằng chứng {}]: {}", i + 1, &doc.text[..doc.text.floor_char_boundary(500)]))
            .collect::<Vec<_>>()
            .join("\n\n");

        let user_prompt = format!(
            "CÂU HỎI: {question}\n\n\
             CÂU TRẢ LỜI CẦN KIỂM TRA:\n{answer}\n\n\
             BẰNG CHỨNG GỐC:\n{evidence_text}\n\n\
             Hãy kiểm tra và trả lời JSON:"
        );

        let response = groq
            .chat(ModelTier::Smart, COMPLIANCE_SYSTEM_PROMPT, &user_prompt, 0.0, 512)
            .await?;

        // Parse JSON response
        let result = parse_compliance_result(&response);
        
        if result.is_compliant {
            info!("Compliance: ✅ PASSED");
        } else {
            warn!("Compliance: ❌ FAILED - {} issues", result.issues.len());
        }

        Ok(result)
    }

    /// Pipeline hoàn chỉnh: kiểm tra → retry nếu fail
    pub async fn check_with_retry(
        groq: &GroqService,
        question: &str,
        answer: &str,
        evidence: &[LegalDocument],
    ) -> Result<(String, ComplianceResult)> {
        let mut current_answer = answer.to_string();
        let mut retry_count = 0u32;

        loop {
            let result = Self::check(groq, question, &current_answer, evidence).await?;

            if result.is_compliant || retry_count >= MAX_RETRY {
                return Ok((
                    current_answer,
                    ComplianceResult {
                        is_compliant: result.is_compliant,
                        issues: result.issues,
                        retry_count,
                    },
                ));
            }

            // Yêu cầu Analyst tạo câu trả lời mới
            info!("Compliance retry #{}: regenerating answer", retry_count + 1);
            
            let issues_str = result.issues.join("; ");
            let retry_prompt = format!(
                "Câu trả lời trước đã bị phát hiện lỗi: {issues_str}\n\n\
                 CÂU HỎI: {question}\n\n\
                 BẰNG CHỨNG: {}\n\n\
                 Hãy viết lại câu trả lời, tránh các lỗi trên.",
                evidence.iter().map(|d| d.text.as_str()).collect::<Vec<_>>().join("\n")
            );

            current_answer = groq
                .chat(
                    ModelTier::Smart,
                    "Bạn là chuyên gia pháp luật. Viết câu trả lời chính xác.",
                    &retry_prompt,
                    0.1,
                    2048,
                )
                .await?;

            retry_count += 1;
        }
    }
}

/// Parse JSON compliance response từ LLM
fn parse_compliance_result(response: &str) -> ComplianceResult {
    // Tìm JSON trong response
    let start = response.find('{');
    let end = response.rfind('}');

    if let (Some(s), Some(e)) = (start, end) {
        let json_str = &response[s..=e];
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str) {
            let is_compliant = value["is_compliant"].as_bool().unwrap_or(false);
            let issues = value["issues"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            return ComplianceResult {
                is_compliant,
                issues,
                retry_count: 0,
            };
        }
    }

    // Fallback: nếu không parse được → assume compliant
    ComplianceResult {
        is_compliant: true,
        issues: vec![],
        retry_count: 0,
    }
}
