//! Document models

use serde::{Deserialize, Serialize};

/// Tài liệu trong Qdrant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalDocument {
    pub text: String,
    pub node_id: i64,
    pub level: i32,
    pub doc_title: String,
    pub doc_id: i64,
    pub score: f64,
}
