//! Cấu hình cho backend server.
//! Đọc biến môi trường từ .env

use anyhow::Result;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub groq_api_key: String,
    pub groq_model: String,
    pub groq_fast_model: String,
    pub qdrant_url: String,
    pub qdrant_collection: String,
    pub rrf_dense_weight: f64,
    pub rrf_sparse_weight: f64,
    pub rrf_k: usize,
    pub host: String,
    pub port: u16,
}

impl AppConfig {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        Ok(Self {
            groq_api_key: std::env::var("GROQ_API_KEY")
                .unwrap_or_else(|_| String::new()),
            groq_model: std::env::var("GROQ_MODEL")
                .unwrap_or_else(|_| "llama-3.3-70b-versatile".to_string()),
            groq_fast_model: std::env::var("GROQ_FAST_MODEL")
                .unwrap_or_else(|_| "llama-3.1-8b-instant".to_string()),
            qdrant_url: std::env::var("QDRANT_URL")
                .unwrap_or_else(|_| "http://localhost:6333".to_string()),
            qdrant_collection: std::env::var("QDRANT_COLLECTION")
                .unwrap_or_else(|_| "legal_raptor".to_string()),
            rrf_dense_weight: std::env::var("RRF_DENSE_WEIGHT")
                .unwrap_or_else(|_| "0.5".to_string())
                .parse()
                .unwrap_or(0.5),
            rrf_sparse_weight: std::env::var("RRF_SPARSE_WEIGHT")
                .unwrap_or_else(|_| "0.5".to_string())
                .parse()
                .unwrap_or(0.5),
            rrf_k: std::env::var("RRF_K")
                .unwrap_or_else(|_| "60".to_string())
                .parse()
                .unwrap_or(60),
            host: std::env::var("HOST")
                .unwrap_or_else(|_| "127.0.0.1".to_string()),
            port: std::env::var("PORT")
                .unwrap_or_else(|_| "8080".to_string())
                .parse()
                .unwrap_or(8080),
        })
    }
}
