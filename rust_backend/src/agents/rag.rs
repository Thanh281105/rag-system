//! RAG Agent - Thực thi HyDE + Hybrid Search + Reranking
//! Lấy bằng chứng pháp lý từ Qdrant

use anyhow::Result;
use tracing::info;

use crate::models::document::LegalDocument;
use crate::services::groq::GroqService;
use crate::services::qdrant::QdrantService;
use crate::services::reranker::RerankerService;

const HYDE_PROMPT: &str = r#"
Bạn là chuyên gia luật Việt Nam. Viết một đoạn văn bản pháp lý ngắn (100-150 từ) 
để trả lời câu hỏi sau. Viết như trích dẫn từ bộ luật thật.
Sử dụng văn phong pháp lý chính thức, trích dẫn số điều khoản nếu có thể.

Câu hỏi: {question}

VĂN BẢN PHÁP LÝ:
"#;

pub struct RagAgent;

impl RagAgent {
    /// Thực thi full RAG pipeline:
    /// 1. HyDE: sinh tài liệu giả định
    /// 2. Tìm kiếm trên Qdrant
    /// 3. Rerank kết quả
    pub async fn retrieve(
        groq: &GroqService,
        qdrant: &QdrantService,
        question: &str,
        top_k: usize,
    ) -> Result<(Vec<LegalDocument>, String)> {
        // 1. HyDE - sinh tài liệu giả định
        let hyde_prompt = HYDE_PROMPT.replace("{question}", question);
        let hyde_document = groq
            .chat(
                "Bạn viết văn bản pháp lý giả định.",
                &hyde_prompt,
                0.7,
                512,
            )
            .await?;

        info!("HyDE document: {} chars", hyde_document.len());

        // 2. Tìm kiếm trên Qdrant
        // Note: Trong production, embedding được tính bằng model local.
        // Ở đây ta dùng text search placeholder vì Rust không có bge-m3.
        // Actual implementation sẽ gọi Python embedding service hoặc ONNX.
        
        // Tạm thời: lấy trực tiếp từ Qdrant bằng dummy vector
        // TODO: Integrate actual embedding model (ONNX/Python service)
        let dummy_vector = vec![0.0f32; 1024]; // Placeholder
        let search_results = qdrant.search_dense(dummy_vector, (top_k * 4) as u64).await?;

        info!("Qdrant search: {} results", search_results.len());

        // 3. Rerank
        let reranked = RerankerService::rerank(
            groq,
            question,
            search_results,
            top_k,
        )
        .await?;

        info!("Reranked: {} documents", reranked.len());

        Ok((reranked, hyde_document))
    }
}
