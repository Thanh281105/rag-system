//! RAG Agent - Thực thi Multi-Query Expansion + Hybrid Search + Reranking
//! Lấy bằng chứng pháp lý từ Qdrant

use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::models::document::LegalDocument;
use crate::services::groq::{GroqService, ModelTier};
use crate::services::qdrant::QdrantService;
use crate::services::reranker::RerankerService;

const MULTI_QUERY_PROMPT: &str = r#"
Bạn là trợ lý pháp lý AI. Nhiệm vụ của bạn là sinh ra 3 cách diễn đạt hoặc khía cạnh khác nhau 
cho cùng một câu hỏi pháp lý, nhằm mở rộng các từ khóa tìm kiếm.
Chỉ trả về đúng 3 câu hỏi, mỗi câu 1 dòng, KHÔNG giải thích, KHÔNG dùng markdown.

Câu hỏi gốc: {question}
"#;

const EMBED_SERVICE_URL: &str = "http://127.0.0.1:8001/embed";

#[derive(Serialize)]
struct EmbedRequest {
    text: String,
}

#[derive(Deserialize)]
struct EmbedResponse {
    vector: Vec<f32>,
}

pub struct RagAgent;

impl RagAgent {
    /// Gọi Python embedding service để lấy vector thật
    async fn get_embedding(text: &str) -> Result<Vec<f32>> {
        let client = Client::new();
        let res = client
            .post(EMBED_SERVICE_URL)
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

    /// Sinh các câu hỏi mở rộng bằng Multi-Query
    pub async fn expand_queries(groq: &GroqService, question: &str) -> Result<String> {
        let prompt = MULTI_QUERY_PROMPT.replace("{question}", question);
        let expanded_queries = groq
            .chat(
                ModelTier::Fast, // Dùng model nhẹ cho expansion để tối ưu tốc độ
                "Bạn sinh ra các câu hỏi phụ.",
                &prompt,
                0.7,
                512,
            )
            .await?;

        info!("Multi-Query expansion generated {} chars", expanded_queries.len());
        Ok(expanded_queries)
    }

    /// Thực thi phần còn lại của RAG pipeline: Tìm kiếm Dense + Sparse (Hybrid) và Rerank
    pub async fn retrieve_with_expanded(
        groq: &GroqService,
        qdrant: &QdrantService,
        question: &str,
        expanded_queries: &str,
        top_k: usize,
    ) -> Result<Vec<LegalDocument>> {
        // 1. Tối đa context: Câu hỏi gốc + Câu hỏi mở rộng
        let combined_query = format!("{}\n\n{}", question, expanded_queries);
        
        // 2. Thực thi song song Dense Search và Sparse Search
        let dense_future = async {
            let query_vector = Self::get_embedding(&combined_query).await?;
            qdrant.search_dense(query_vector, (top_k * 4) as u64).await
        };
        let sparse_future = qdrant.search_sparse(&combined_query, (top_k * 4) as u64);

        let (dense_res, sparse_res) = tokio::join!(dense_future, sparse_future);
        
        let dense_docs = dense_res?;
        let sparse_docs = sparse_res?;

        info!("Dense: {} results, Sparse: {} results", dense_docs.len(), sparse_docs.len());

        // 3. Reciprocal Rank Fusion (RRF)
        let mut rrf_scores: std::collections::HashMap<i64, f64> = std::collections::HashMap::new();
        let mut doc_map: std::collections::HashMap<i64, LegalDocument> = std::collections::HashMap::new();

        let k_rrf = 60.0;

        for (rank, doc) in dense_docs.into_iter().enumerate() {
            let score = 1.0 / (k_rrf + (rank as f64) + 1.0);
            *rrf_scores.entry(doc.node_id).or_insert(0.0) += score;
            doc_map.insert(doc.node_id, doc);
        }

        for (rank, doc) in sparse_docs.into_iter().enumerate() {
            let score = 1.0 / (k_rrf + (rank as f64) + 1.0);
            *rrf_scores.entry(doc.node_id).or_insert(0.0) += score;
            doc_map.entry(doc.node_id).or_insert(doc);
        }

        let mut fused_results: Vec<LegalDocument> = doc_map.into_values().collect();
        // Sort descending by RRF score stored in `rrf_scores`
        fused_results.sort_by(|a, b| {
            let score_a = rrf_scores.get(&a.node_id).unwrap_or(&0.0);
            let score_b = rrf_scores.get(&b.node_id).unwrap_or(&0.0);
            score_b.partial_cmp(score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Lấy top_k * 2 để gửi cho Reranker (nhiều hơn một chút để reranker chọn lọc)
        let rrf_top = fused_results.into_iter().take(top_k * 2).collect::<Vec<_>>();

        info!("RRF combined: {} unique results, returning top {}", rrf_scores.len(), rrf_top.len());

        // 4. Rerank
        let reranked = RerankerService::rerank(groq, question, rrf_top, top_k).await?;

        info!("Reranked: {} documents", reranked.len());

        Ok(reranked)
    }
}
