//! RAG-Router Agent (Agent 1)
//! Kết hợp: Phân loại ý định + Translate VN→EN + Multi-Query Expansion + Retrieval
//! Đây là agent đầu tiên trong pipeline 3-agent

use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::models::document::ArxivDocument;
use crate::services::groq::{GroqService, ModelTier};
use crate::services::qdrant::QdrantService;
use crate::services::reranker::RerankerService;

// ─── Intent Classification ──────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum QueryIntent {
    /// Câu hỏi kỹ thuật/khoa học → cần RAG pipeline
    Technical,
    /// Giao tiếp thông thường → trả lời trực tiếp
    Casual,
}

const ROUTER_SYSTEM_PROMPT: &str = r#"
Bạn là Router Agent trong hệ thống trả lời câu hỏi AI/ML research.
Nhiệm vụ: phân loại ý định câu hỏi.

Trả lời ĐÚNG MỘT từ:
- "TECHNICAL" nếu câu hỏi liên quan đến AI, ML, Deep Learning, NLP, Computer Vision, toán học, thuật toán, papers nghiên cứu
- "CASUAL" nếu là giao tiếp thông thường, chào hỏi, hoặc không liên quan kỹ thuật

Chỉ trả lời TECHNICAL hoặc CASUAL, không giải thích.
"#;

pub struct RagRouterAgent;

impl RagRouterAgent {
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

        let intent = if response.trim().to_uppercase().contains("TECHNICAL") {
            QueryIntent::Technical
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
            "Bạn là trợ lý nghiên cứu AI thân thiện. Trả lời ngắn gọn, lịch sự bằng tiếng Việt. \
             Nếu người dùng hỏi về AI/ML, khuyên họ hỏi cụ thể hơn.",
            question,
            0.7,
            256,
        )
        .await
    }

    // ─── Translation VN → EN ────────────────────────────────────────

    /// Dịch câu hỏi từ tiếng Việt sang tiếng Anh
    pub async fn translate_to_english(groq: &GroqService, question_vn: &str) -> Result<String> {
        let prompt = format!(
            "Translate the following Vietnamese question to English. \
             Keep all technical terms (RAG, Transformer, LoRA, RLHF, etc.) unchanged.\n\n\
             Vietnamese: {question_vn}\n\nEnglish:"
        );

        let translated = groq
            .chat(
                ModelTier::Fast,
                "You are a precise translator. Translate Vietnamese to English. Keep technical terms unchanged. Only output the translation, nothing else.",
                &prompt,
                0.0,
                256,
            )
            .await?;

        info!("Translated: '{}' → '{}'", &question_vn[..question_vn.floor_char_boundary(40)], &translated[..translated.floor_char_boundary(40)]);
        Ok(translated.trim().to_string())
    }

    // ─── Multi-Query Expansion (English) ────────────────────────────

    /// Sinh 3 biến thể tiếng Anh cho câu hỏi
    pub async fn expand_queries_en(groq: &GroqService, question_en: &str) -> Result<String> {
        let prompt = format!(
            "Generate exactly 3 different English search queries for the same research question.\n\
             Each query should take a different angle:\n\
             1. Semantic rephrase (different wording, same meaning)\n\
             2. Keyword-focused (extract key technical terms)\n\
             3. Technical phrasing (formal academic style)\n\n\
             Original question: {question_en}\n\n\
             Return only 3 queries, one per line, no numbering, no explanation."
        );

        let expanded = groq
            .chat(
                ModelTier::Fast,
                "You generate search queries for academic paper retrieval.",
                &prompt,
                0.7,
                512,
            )
            .await?;

        info!("Multi-Query expansion: {} chars", expanded.len());
        Ok(expanded)
    }

    // ─── Retrieval Pipeline ─────────────────────────────────────────

    /// Gọi Python embedding service để lấy vector
    async fn get_embedding(text: &str) -> Result<Vec<f32>> {
        let client = Client::new();
        let res = client
            .post("http://127.0.0.1:8001/embed")
            .json(&EmbedRequest {
                text: text.to_string(),
            })
            .send()
            .await?;

        if !res.status().is_success() {
            let err_text = res.text().await.unwrap_or_default();
            anyhow::bail!("Embedding service error: {}", err_text);
        }

        let embed_res: EmbedResponse = res.json().await?;
        Ok(embed_res.vector)
    }

    /// Thực thi retrieval pipeline: Hybrid Search + Rerank
    pub async fn retrieve(
        groq: &GroqService,
        qdrant: &QdrantService,
        question_en: &str,
        expanded_queries: &str,
        top_k: usize,
    ) -> Result<Vec<ArxivDocument>> {
        // Kết hợp query gốc + expanded queries
        let combined_query = format!("{}\n\n{}", question_en, expanded_queries);

        // Song song: Dense Search + Sparse Search
        let dense_future = async {
            let query_vector = Self::get_embedding(&combined_query).await?;
            qdrant.search_dense(query_vector, (top_k * 4) as u64).await
        };
        let sparse_future = qdrant.search_sparse(&combined_query, (top_k * 4) as u64);

        let (dense_res, sparse_res) = tokio::join!(dense_future, sparse_future);

        let dense_docs = dense_res?;
        let sparse_docs = sparse_res?;

        info!("Dense: {} results, Sparse: {} results", dense_docs.len(), sparse_docs.len());

        // RRF (Reciprocal Rank Fusion)
        let config = crate::config::AppConfig::from_env().unwrap_or_else(|_| crate::config::AppConfig {
            groq_api_key: String::new(),
            groq_model: String::new(),
            groq_fast_model: String::new(),
            qdrant_url: String::new(),
            qdrant_collection: String::new(),
            rrf_dense_weight: 0.6,
            rrf_sparse_weight: 0.4,
            rrf_k: 60,
            host: String::new(),
            port: 8080,
        });

        let mut rrf_scores: std::collections::HashMap<i64, f64> = std::collections::HashMap::new();
        let mut doc_map: std::collections::HashMap<i64, ArxivDocument> = std::collections::HashMap::new();

        let k_rrf = config.rrf_k as f64;

        for (rank, doc) in dense_docs.into_iter().enumerate() {
            let score = config.rrf_dense_weight * (1.0 / (k_rrf + (rank as f64) + 1.0));
            *rrf_scores.entry(doc.node_id).or_insert(0.0) += score;
            doc_map.insert(doc.node_id, doc);
        }

        for (rank, doc) in sparse_docs.into_iter().enumerate() {
            let score = config.rrf_sparse_weight * (1.0 / (k_rrf + (rank as f64) + 1.0));
            *rrf_scores.entry(doc.node_id).or_insert(0.0) += score;
            doc_map.entry(doc.node_id).or_insert(doc);
        }

        let mut fused_results: Vec<ArxivDocument> = doc_map.into_values().collect();
        fused_results.sort_by(|a, b| {
            let score_a = rrf_scores.get(&a.node_id).unwrap_or(&0.0);
            let score_b = rrf_scores.get(&b.node_id).unwrap_or(&0.0);
            score_b.partial_cmp(score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        let rrf_top = fused_results.into_iter().take(top_k * 2).collect::<Vec<_>>();
        info!("RRF combined: {} unique results", rrf_scores.len());

        // Rerank
        let reranked = RerankerService::rerank(groq, question_en, rrf_top, top_k).await?;
        info!("Reranked: {} documents", reranked.len());

        Ok(reranked)
    }
}

#[derive(Serialize)]
struct EmbedRequest {
    text: String,
}

#[derive(Deserialize)]
struct EmbedResponse {
    vector: Vec<f32>,
}
