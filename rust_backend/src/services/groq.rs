//! Groq API service - gọi LLM cho các agent

use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::config::AppConfig;

#[derive(Clone)]
pub struct GroqService {
    client: Client,
    api_key: String,
    model: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f64,
    max_tokens: u32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

impl GroqService {
    pub fn new(config: &AppConfig) -> Self {
        Self {
            client: Client::new(),
            api_key: config.groq_api_key.clone(),
            model: config.groq_model.clone(),
        }
    }

    /// Gọi Groq API với system prompt và user message
    pub async fn chat(
        &self,
        system_prompt: &str,
        user_message: &str,
        temperature: f64,
        max_tokens: u32,
    ) -> Result<String> {
        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_message.to_string(),
                },
            ],
            temperature,
            max_tokens,
        };

        let response = self
            .client
            .post("https://api.groq.com/openai/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        let response_text = response.text().await?;
        let chat_response: ChatResponse = serde_json::from_str(&response_text)?;

        let content = chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        info!("Groq response: {} chars", content.len());
        Ok(content)
    }

    /// Chat với messages tùy chỉnh
    pub async fn chat_with_messages(
        &self,
        messages: Vec<ChatMessage>,
        temperature: f64,
        max_tokens: u32,
    ) -> Result<String> {
        let request = ChatRequest {
            model: self.model.clone(),
            messages,
            temperature,
            max_tokens,
        };

        let response = self
            .client
            .post("https://api.groq.com/openai/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        let chat_response: ChatResponse = response.json().await?;

        Ok(chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default())
    }
}
