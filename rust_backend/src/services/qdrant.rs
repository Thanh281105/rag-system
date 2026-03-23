//! Qdrant service - truy vấn vector database

use anyhow::Result;
use md5::{Digest, Md5};
use qdrant_client::qdrant::{QueryPointsBuilder, SparseVector, Value};
use qdrant_client::Qdrant;
use std::collections::HashMap;
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

    /// Vietnamese stop-words - các từ phổ biến không mang ý nghĩa pháp lý
    /// Loại bỏ để sparse vector tập trung vào keywords quan trọng
    const VIETNAMESE_STOP_WORDS: &'static [&'static str] = &[
        // Giới từ / Liên từ
        "của", "và", "các", "có", "được", "trong", "là", "cho", "với", "về",
        "theo", "từ", "đến", "tại", "bởi", "do", "khi", "nếu", "thì", "mà",
        "hoặc", "hay", "cũng", "đã", "sẽ", "đang", "để", "bị", "không",
        // Đại từ
        "này", "đó", "những", "một", "các", "mỗi", "mọi", "tất", "cả",
        // Trợ từ / Phụ từ
        "rằng", "thì", "lại", "vẫn", "còn", "nên", "phải", "cần",
        "như", "trên", "dưới", "trước", "sau", "giữa", "ngoài",
        // Từ nối phổ biến trong văn bản luật (nhưng không mang thông tin pháp lý)
        "quy", "định", "việc", "người", "nhưng", "vào", "ra",
    ];

    /// Helper tạo vector thưa (sparse vector) tương thích 100% với Python (MD5 modulo 100k).
    /// Cải tiến: loại bỏ stop-words tiếng Việt + log-scaled TF (gần BM25 hơn).
    fn text_to_sparse(text: &str) -> (Vec<u32>, Vec<f32>) {
        let mut word_freq: HashMap<u32, f32> = HashMap::new();
        let words = text.to_lowercase();

        // Xây dựng HashSet stop-words để tra cứu O(1)
        let stop_set: std::collections::HashSet<&str> =
            Self::VIETNAMESE_STOP_WORDS.iter().copied().collect();

        for w in words.split_whitespace() {
            // Bỏ qua stop-words và từ quá ngắn (1 ký tự)
            if stop_set.contains(w) || w.chars().count() <= 1 {
                continue;
            }

            let mut hasher = Md5::new();
            hasher.update(w.as_bytes());
            let result = hasher.finalize();
            let hex_str = hex::encode(result);

            if let Ok(big_int) = u128::from_str_radix(&hex_str, 16) {
                let h = (big_int % 100000) as u32;
                *word_freq.entry(h).or_insert(0.0) += 1.0;
            }
        }

        let mut indices = Vec::new();
        let mut values = Vec::new();
        for (k, raw_tf) in word_freq {
            indices.push(k);
            // Log-scaled TF: 1 + ln(tf) — giảm ảnh hưởng của từ lặp quá nhiều
            values.push(1.0 + raw_tf.ln());
        }

        (indices, values)
    }

    /// Tìm kiếm keyword (sparse vector / BM25-like)
    pub async fn search_sparse(
        &self,
        query_text: &str,
        top_k: u64,
    ) -> Result<Vec<LegalDocument>> {
        let (indices, values) = Self::text_to_sparse(query_text);

        let results = self
            .client
            .query(
                QueryPointsBuilder::new(&self.collection)
                    .query(qdrant_client::qdrant::Query::from(
                        qdrant_client::qdrant::VectorInput::new_sparse(indices, values),
                    ))
                    .using("sparse")
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

        info!("Qdrant sparse search: {} results", documents.len());
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
