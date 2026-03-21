//! Qdrant service - truy vấn vector database

use anyhow::Result;
use qdrant_client::qdrant::{
    QueryPointsBuilder, Value,
};
use qdrant_client::Qdrant;
use tracing::info;

use crate::config::AppConfig;
use crate::models::document::LegalDocument;

#[derive(Clone)]
pub struct QdrantService {
    client: Qdrant,
    collection: String,
}

impl QdrantService {
    pub async fn new(config: &AppConfig) -> Result<Self> {
        let client = Qdrant::from_url(&config.qdrant_url).build()?;
        info!("Connected to Qdrant: {}", config.qdrant_url);

        Ok(Self {
            client,
            collection: config.qdrant_collection.clone(),
        })
    }

    /// Tìm kiếm semantic (dense vector)
    pub async fn search_dense(
        &self,
        query_vector: Vec<f32>,
        top_k: u64,
    ) -> Result<Vec<LegalDocument>> {
        let results = self
            .client
            .query(
                QueryPointsBuilder::new(&self.collection)
                    .query(qdrant_client::qdrant::Query::from(
                        qdrant_client::qdrant::VectorInput::from(query_vector),
                    ))
                    .using("dense")
                    .limit(top_k)
                    .with_payload(true),
            )
            .await?;

        let documents: Vec<LegalDocument> = results
            .result
            .into_iter()
            .map(|point| {
                let payload = point.payload;
                LegalDocument {
                    text: extract_string(&payload, "text"),
                    node_id: extract_integer(&payload, "node_id"),
                    level: extract_integer(&payload, "level") as i32,
                    doc_title: extract_string(&payload, "doc_title"),
                    doc_id: extract_integer(&payload, "doc_id"),
                    score: point.score as f64,
                }
            })
            .collect();

        info!("Qdrant dense search: {} results", documents.len());
        Ok(documents)
    }

    /// Kiểm tra kết nối
    pub async fn health_check(&self) -> Result<bool> {
        let info = self
            .client
            .collection_info(&self.collection)
            .await?;
        Ok(info.result.is_some())
    }
}

/// Helper: extract string from Qdrant payload
fn extract_string(
    payload: &std::collections::HashMap<String, Value>,
    key: &str,
) -> String {
    payload
        .get(key)
        .and_then(|v| match &v.kind {
            Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

/// Helper: extract integer from Qdrant payload
fn extract_integer(
    payload: &std::collections::HashMap<String, Value>,
    key: &str,
) -> i64 {
    payload
        .get(key)
        .and_then(|v| match &v.kind {
            Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)) => Some(*i),
            _ => None,
        })
        .unwrap_or(0)
}
