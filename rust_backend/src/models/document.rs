//! Document models

use serde::{Deserialize, Serialize};

/// Tài liệu ArXiv trong Qdrant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArxivDocument {
    pub text: String,
    pub node_id: i64,
    pub level: i32,
    pub doc_title: String,
    pub doc_id: i64,
    pub score: f64,
    pub year: i32,
    pub authors: String,
    pub arxiv_id: String,
}
