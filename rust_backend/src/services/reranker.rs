//! Reranking service - sử dụng Python Cross-encoder endpoint
//! Gọi bge-reranker-v2-m3 thật qua microservice.

use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::models::document::ArxivDocument;
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
    /// Rerank documents bằng Cross-encoder thật.
    /// Fallback sang LLM-based scoring nếu service không khả dụng.
    pub async fn rerank(
        groq: &GroqService,
        query: &str,
        documents: Vec<ArxivDocument>,
        top_k: usize,
    ) -> Result<Vec<ArxivDocument>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        match Self::rerank_via_cross_encoder(query, &documents, top_k).await {
            Ok(reranked) => {
                info!("✅ Cross-encoder reranked: {} → {} documents", documents.len(), reranked.len());
                return Ok(reranked);
            }
            Err(e) => {
                warn!("⚠️ Cross-encoder service unavailable ({}), fallback to LLM reranking", e);
            }
        }

        Self::rerank_via_llm(groq, query, documents, top_k).await
    }

    async fn rerank_via_cross_encoder(
        query: &str,
        documents: &[ArxivDocument],
        top_k: usize,
    ) -> Result<Vec<ArxivDocument>> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

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

        let reranked: Vec<ArxivDocument> = rerank_res
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

    async fn rerank_via_llm(
        groq: &GroqService,
        query: &str,
        documents: Vec<ArxivDocument>,
        top_k: usize,
    ) -> Result<Vec<ArxivDocument>> {
        let mut scored_docs: Vec<(f64, ArxivDocument)> = Vec::new();

        for doc in &documents {
            let prompt = format!(
                "Rate relevance (0.0-1.0) of this text to the query.\n\n\
                 Query: {query}\n\n\
                 Text: {}\n\n\
                 Reply with ONLY a decimal number (e.g., 0.85):",
                &doc.text[..doc.text.floor_char_boundary(500)]
            );

            let score_str = groq
                .chat(ModelTier::Smart, "You are a scoring system. Only reply with a number.", &prompt, 0.0, 10)
                .await
                .unwrap_or_default();

            let score = score_str.trim().parse::<f64>().unwrap_or(0.0);
            scored_docs.push((score, doc.clone()));
        }

        scored_docs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let reranked: Vec<ArxivDocument> = scored_docs
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
