//! Qdrant service - truy vấn vector database

use anyhow::Result;
use qdrant_client::Qdrant;
use tracing::info;

use crate::config::AppConfig;

#[derive(Clone)]
pub struct QdrantService {
    client: Qdrant,
    collection: String,
}

impl QdrantService {
    pub async fn new(config: &AppConfig) -> Result<Self> {
        let grpc_url = config.qdrant_url.replace(":6333", ":6334");
        let client = Qdrant::from_url(&grpc_url)
            .skip_compatibility_check()
            .build()?;
        info!("Connected to Qdrant: {}", grpc_url);

        Ok(Self {
            client,
            collection: config.qdrant_collection.clone(),
        })
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
