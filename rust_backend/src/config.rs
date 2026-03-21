//! Cấu hình cho backend server.
//! Đọc biến môi trường từ .env

use anyhow::Result;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub groq_api_key: String,
    pub groq_model: String,
    pub qdrant_url: String,
    pub qdrant_collection: String,
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
            qdrant_url: std::env::var("QDRANT_URL")
                .unwrap_or_else(|_| "http://localhost:6333".to_string()),
            qdrant_collection: std::env::var("QDRANT_COLLECTION")
                .unwrap_or_else(|_| "legal_raptor".to_string()),
            host: std::env::var("HOST")
                .unwrap_or_else(|_| "127.0.0.1".to_string()),
            port: std::env::var("PORT")
                .unwrap_or_else(|_| "8080".to_string())
                .parse()
                .unwrap_or(8080),
        })
    }
}
