//! Request/Response models cho API

use serde::{Deserialize, Serialize};

/// Tin nhắn trong lịch sử hội thoại
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatHistoryMessage {
    pub role: String,   // "user" hoặc "assistant"
    pub content: String,
}

/// Request gửi câu hỏi
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
    pub authors: String,
    pub year: i32,
    pub arxiv_id: String,
    pub relevance_score: f64,
    pub level: i32,
}

/// Trace luồng xử lý của các agent
#[derive(Debug, Serialize)]
pub struct AgentTrace {
    pub router_decision: String,
    pub translated_query: String,
    pub expanded_queries: String,
    pub retrieved_count: usize,
    pub reranked_count: usize,
    pub analyst_answer: String,
    pub reviewer_triggered: bool,
    pub reviewer_result: ReviewerResult,
}

/// Kết quả kiểm tra của Reviewer Agent
#[derive(Debug, Serialize)]
pub struct ReviewerResult {
    pub is_approved: bool,
    pub issues: Vec<String>,
    pub retry_count: u32,
}
