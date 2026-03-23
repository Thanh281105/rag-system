//! Request/Response models cho API

use serde::{Deserialize, Serialize};

/// Tin nhắn trong lịch sử hội thoại
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatHistoryMessage {
    pub role: String,   // "user" hoặc "assistant"
    pub content: String,
}

/// Request gửi câu hỏi pháp lý
#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    pub question: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub history: Vec<ChatHistoryMessage>,
}

fn default_top_k() -> usize {
    5
}

/// Response trả lời câu hỏi
#[derive(Debug, Serialize)]
pub struct QueryResponse {
    pub answer: String,
    pub sources: Vec<SourceDocument>,
    pub agent_trace: AgentTrace,
    pub processing_time_ms: u128,
}

/// Tài liệu nguồn được trích dẫn
#[derive(Debug, Serialize, Clone)]
pub struct SourceDocument {
    pub text: String,
    pub doc_title: String,
    pub relevance_score: f64,
    pub level: i32,
}

/// Trace luồng xử lý của các agent
#[derive(Debug, Serialize)]
pub struct AgentTrace {
    pub router_decision: String,
    pub hyde_document: String,
    pub retrieved_count: usize,
    pub reranked_count: usize,
    pub analyst_reasoning: String,
    pub compliance_check: ComplianceResult,
}

/// Kết quả kiểm tra của Compliance Agent
#[derive(Debug, Serialize)]
pub struct ComplianceResult {
    pub is_compliant: bool,
    pub issues: Vec<String>,
    pub retry_count: u32,
}
