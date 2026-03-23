//! Reranking service - sử dụng Python Cross-encoder endpoint
//! Gọi bge-reranker-v2-m3 thật qua microservice thay vì LLM scoring.
//! Kết quả chính xác hơn VÀ nhanh hơn (1 batch call thay vì N API calls).

use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::models::document::LegalDocument;
use crate::services::groq::{GroqService, ModelTier};

const RERANK_SERVICE_URL: &str = "http://127.0.0.1:8001/rerank";

#[derive(Serialize)]
struct RerankRequest {
    query: String,
    documents: Vec<String>,
    top_k: usize,
}

#[derive(Deserialize)]
struct RerankResult {
    index: usize,
    score: f64,
}

#[derive(Deserialize)]
struct RerankResponse {
    results: Vec<RerankResult>,
}

pub struct RerankerService;

impl RerankerService {
    /// Rerank documents bằng Cross-encoder thật (gọi Python service).
    /// Fallback sang LLM-based scoring nếu service không khả dụng.
    pub async fn rerank(
        groq: &GroqService,
        query: &str,
        documents: Vec<LegalDocument>,
        top_k: usize,
    ) -> Result<Vec<LegalDocument>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        // Thử gọi Python cross-encoder service trước
        match Self::rerank_via_cross_encoder(query, &documents, top_k).await {
            Ok(reranked) => {
                info!("✅ Cross-encoder reranked: {} → {} documents", documents.len(), reranked.len());
                return Ok(reranked);
            }
            Err(e) => {
                warn!("⚠️ Cross-encoder service không khả dụng ({}), fallback sang LLM reranking", e);
            }
        }

        // Fallback: LLM-based scoring (chậm hơn nhưng luôn khả dụng)
        Self::rerank_via_llm(groq, query, documents, top_k).await
    }

    /// Cross-encoder reranking qua Python microservice (nhanh + chính xác)
    async fn rerank_via_cross_encoder(
        query: &str,
        documents: &[LegalDocument],
        top_k: usize,
    ) -> Result<Vec<LegalDocument>> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        // Chuẩn bị danh sách text (cắt 500 ký tự để tối ưu tốc độ)
        let doc_texts: Vec<String> = documents
            .iter()
            .map(|doc| doc.text[..doc.text.floor_char_boundary(500)].to_string())
            .collect();

        let request = RerankRequest {
            query: query.to_string(),
            documents: doc_texts,
            top_k,
        };

        let response = client
            .post(RERANK_SERVICE_URL)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let err_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Rerank service error: {}", err_text);
        }

        let rerank_res: RerankResponse = response.json().await?;

        // Map kết quả về LegalDocument với score mới
        let reranked: Vec<LegalDocument> = rerank_res
            .results
            .into_iter()
            .filter_map(|r| {
                documents.get(r.index).map(|doc| {
                    let mut new_doc = doc.clone();
                    new_doc.score = r.score;
                    new_doc
                })
            })
            .collect();

        Ok(reranked)
    }

    /// LLM-based reranking (fallback, chậm hơn)
    async fn rerank_via_llm(
        groq: &GroqService,
        query: &str,
        documents: Vec<LegalDocument>,
        top_k: usize,
    ) -> Result<Vec<LegalDocument>> {
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

        scored_docs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let reranked: Vec<LegalDocument> = scored_docs
            .into_iter()
            .take(top_k)
            .map(|(score, mut doc)| {
                doc.score = score;
                doc
            })
            .collect();

        info!("LLM Reranked: {} → {} documents", documents.len(), reranked.len());
        Ok(reranked)
    }
}
