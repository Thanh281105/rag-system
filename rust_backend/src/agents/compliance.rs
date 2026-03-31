//! Reviewer Agent (Agent 3) - Conditional fact-checking
//! Chỉ chạy khi cần: query hỏi numbers/metrics/formulas hoặc low confidence
//! So sánh Answer (VN) với Source (EN) để detect hallucination

use anyhow::Result;
use tracing::{info, warn};

use crate::models::document::ArxivDocument;
use crate::models::query::ReviewerResult;
use crate::services::groq::{GroqService, ModelTier};

const REVIEWER_SYSTEM_PROMPT: &str = r#"
You are a Reviewer Agent — a strict fact-checker for cross-lingual AI research Q&A.

TASK: Compare the Vietnamese ANSWER against the English SOURCE EVIDENCE and detect:
1. Hallucination — information NOT present in the evidence
2. Incorrect translation — technical terms wrongly translated
3. Wrong numbers — accuracy figures, parameter counts, dates that don't match
4. Unsupported conclusions — claims not backed by evidence
5. Shallow answer — answer only paraphrases the paper title without providing methodology, specific results, or numbers from the evidence
6. Missing details — evidence contains specific metrics/results but answer omits them

IMPORTANT REJECTION CRITERIA:
- If the answer ONLY restates what the paper is about (paraphrasing the title) without citing specific methodology, experiments, or numerical results from the evidence → REJECT
- If the evidence contains specific numbers (accuracy, F1, parameters, etc.) but the answer does not mention any → REJECT
- If the answer makes claims that go beyond what the evidence supports → REJECT

RESPOND in JSON format:
{
    "is_approved": true/false,
    "issues": ["Issue 1", "Issue 2"],
    "suggestion": "Brief suggestion if there are issues"
}
"#;

const MAX_RETRY: u32 = 2;

pub struct ReviewerAgent;

impl ReviewerAgent {
    /// Kiểm tra câu trả lời
    pub async fn check(
        groq: &GroqService,
        question: &str,
        answer: &str,
        evidence: &[ArxivDocument],
    ) -> Result<ReviewerResult> {
        let evidence_text: String = evidence
            .iter()
            .enumerate()
            .map(|(i, doc)| format!("[Evidence {}]: {}", i + 1, &doc.text[..doc.text.floor_char_boundary(1500)]))
            .collect::<Vec<_>>()
            .join("\n\n");

        let user_prompt = format!(
            "QUESTION: {question}\n\n\
             ANSWER TO CHECK (Vietnamese):\n{answer}\n\n\
             SOURCE EVIDENCE (English):\n{evidence_text}\n\n\
             Check and respond in JSON:"
        );

        let response = groq
            .chat(ModelTier::Smart, REVIEWER_SYSTEM_PROMPT, &user_prompt, 0.0, 512)
            .await?;

        let result = parse_reviewer_result(&response);

        if result.is_approved {
            info!("Reviewer: ✅ APPROVED");
        } else {
            warn!("Reviewer: ❌ REJECTED - {} issues", result.issues.len());
        }

        Ok(result)
    }

    /// Pipeline: kiểm tra → retry nếu fail
    pub async fn check_with_retry(
        groq: &GroqService,
        question: &str,
        answer: &str,
        evidence: &[ArxivDocument],
    ) -> Result<(String, ReviewerResult)> {
        let mut current_answer = answer.to_string();
        let mut retry_count = 0u32;

        loop {
            let result = Self::check(groq, question, &current_answer, evidence).await?;

            if result.is_approved || retry_count >= MAX_RETRY {
                return Ok((
                    current_answer,
                    ReviewerResult {
                        is_approved: result.is_approved,
                        issues: result.issues,
                        retry_count,
                    },
                ));
            }

            info!("Reviewer retry #{}: regenerating answer", retry_count + 1);

            let issues_str = result.issues.join("; ");
            let retry_prompt = format!(
                "Câu trả lời trước đã bị phát hiện lỗi: {issues_str}\n\n\
                 CÂU HỎI: {question}\n\n\
                 EVIDENCE: {}\n\n\
                 Hãy viết lại câu trả lời bằng tiếng Việt, tránh các lỗi trên. \
                 Giữ thuật ngữ kỹ thuật bằng tiếng Anh.",
                evidence.iter().map(|d| d.text.as_str()).collect::<Vec<_>>().join("\n")
            );

            current_answer = groq
                .chat(
                    ModelTier::Smart,
                    "Bạn là chuyên gia AI. Viết câu trả lời chính xác bằng tiếng Việt.",
                    &retry_prompt,
                    0.1,
                    2048,
                )
                .await?;

            retry_count += 1;
        }
    }
}

/// Parse JSON reviewer response từ LLM
fn parse_reviewer_result(response: &str) -> ReviewerResult {
    let start = response.find('{');
    let end = response.rfind('}');

    if let (Some(s), Some(e)) = (start, end) {
        let json_str = &response[s..=e];
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str) {
            let is_approved = value["is_approved"].as_bool().unwrap_or(false);
            let issues = value["issues"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            return ReviewerResult {
                is_approved,
                issues,
                retry_count: 0,
            };
        }
    }

    // Fallback: assume approved if can't parse
    ReviewerResult {
        is_approved: true,
        issues: vec![],
        retry_count: 0,
    }
}
