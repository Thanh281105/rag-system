//! Reranking service (placeholder - sử dụng LLM-based reranking)
//! Trong production, nên dùng cross-encoder local, nhưng ở đây dùng Groq API
//! để đơn giản hoá deployment.

use anyhow::Result;
use tracing::info;

use crate::models::document::LegalDocument;
use crate::services::groq::{GroqService, ModelTier};

pub struct RerankerService;

impl RerankerService {
    /// Rerank documents bằng LLM scoring
    pub async fn rerank(
        groq: &GroqService,
        query: &str,
        documents: Vec<LegalDocument>,
        top_k: usize,
    ) -> Result<Vec<LegalDocument>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        // Với mỗi document, tạo prompt và score
        let mut scored_docs: Vec<(f64, LegalDocument)> = Vec::new();

        for doc in &documents {
            let prompt = format!(
                "Đánh giá mức độ liên quan (0.0-1.0) của đoạn văn bản sau với câu hỏi.\n\n\
                 Câu hỏi: {query}\n\n\
                 Văn bản: {}\n\n\
                 CHỉ trả lời MỘT số thập phân (ví dụ: 0.85):",
                &doc.text[..doc.text.floor_char_boundary(500)]
            );

            let score_str = groq
                .chat(
                    ModelTier::Smart,
                    "Bạn là hệ thống scoring. Chỉ trả lời một số.",
                    &prompt,
                    0.0,
                    10,
                )
                .await
                .unwrap_or_default();

            let score = score_str.trim().parse::<f64>().unwrap_or(0.0);
            scored_docs.push((score, doc.clone()));
        }

        // Sort by rerank score descending
        scored_docs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let reranked: Vec<LegalDocument> = scored_docs
            .into_iter()
            .take(top_k)
            .map(|(score, mut doc)| {
                doc.score = score;
                doc
            })
            .collect();

        info!("Reranked: {} → {} documents", documents.len(), reranked.len());
        Ok(reranked)
    }
}
