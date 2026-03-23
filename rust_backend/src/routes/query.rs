//! Query endpoint - Luồng xử lý chính của hệ thống multi-agent

use actix_web::{post, web, HttpResponse};
use std::time::Instant;
use tracing::info;

use crate::agents::analyst::AnalystAgent;
use crate::agents::compliance::ComplianceAgent;
use crate::agents::rag::RagAgent;
use crate::agents::router::{QueryIntent, RouterAgent};
use crate::models::query::{
    AgentTrace, ChatHistoryMessage, ComplianceResult, QueryRequest, QueryResponse, SourceDocument,
};
use crate::services::groq::GroqService;
use crate::services::qdrant::QdrantService;

#[post("/api/query")]
pub async fn handle_query(
    body: web::Json<QueryRequest>,
    groq: web::Data<GroqService>,
    qdrant: web::Data<QdrantService>,
) -> HttpResponse {
    let start = Instant::now();
    let question = &body.question;
    let top_k = body.top_k;
    let history = &body.history;

    info!("📨 Query: '{}' (history: {} msgs)", &question[..question.floor_char_boundary(80)], history.len());

    // Xây dựng ngữ cảnh hội thoại để bổ sung cho câu hỏi
    let contextualized_question = build_contextualized_question(question, history);

    // ─── Step 1: Tối ưu hoá (Parallel Execution) ──────────────────────
    // Chạy song song Router (phân loại câu hỏi) & RagAgent (mở rộng câu hỏi)
    let intent_future = RouterAgent::classify(&groq, &contextualized_question);
    let expansion_future = RagAgent::expand_queries(&groq, &contextualized_question);

    let (intent_res, expansion_res) = tokio::join!(intent_future, expansion_future);

    let intent = match intent_res {
        Ok(i) => i,
        Err(e) => {
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("Router error: {e}")
            }));
        }
    };

    let expanded_queries = match expansion_res {
        Ok(eq) => eq,
        Err(e) => {
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("Multi-Query Expansion error: {e}")
            }));
        }
    };

    // Casual → trả lời trực tiếp
    if intent == QueryIntent::Casual {
        let casual_answer = RouterAgent::casual_response(&groq, &contextualized_question)
            .await
            .unwrap_or_else(|_| "Xin lỗi, tôi không thể xử lý yêu cầu này.".to_string());

        return HttpResponse::Ok().json(QueryResponse {
            answer: casual_answer,
            sources: vec![],
            agent_trace: AgentTrace {
                router_decision: "CASUAL".to_string(),
                hyde_document: String::new(),
                retrieved_count: 0,
                reranked_count: 0,
                analyst_reasoning: String::new(),
                compliance_check: ComplianceResult {
                    is_compliant: true,
                    issues: vec![],
                    retry_count: 0,
                },
            },
            processing_time_ms: start.elapsed().as_millis(),
        });
    }

    // ─── Step 2: RAG Agent (Retrieval) ───────────────────────────────
    let evidence = match RagAgent::retrieve_with_expanded(
        &groq, 
        &qdrant, 
        &contextualized_question, 
        &expanded_queries, 
        top_k
    ).await {
        Ok(result) => result,
        Err(e) => {
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("RAG error: {e}")
            }));
        }
    };

    // ─── Step 3: Analyst Agent ───────────────────────────
    let analyst_answer = match AnalystAgent::analyze(&groq, &contextualized_question, &evidence).await {
        Ok(answer) => answer,
        Err(e) => {
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("Analyst error: {e}")
            }));
        }
    };

    // ─── Step 4: Compliance Agent ────────────────────────
    let (final_answer, compliance_result) =
        match ComplianceAgent::check_with_retry(&groq, question, &analyst_answer, &evidence).await
        {
            Ok(result) => result,
            Err(e) => {
                // Nếu compliance fail → vẫn trả về answer gốc
                (
                    analyst_answer.clone(),
                    ComplianceResult {
                        is_compliant: false,
                        issues: vec![format!("Compliance check error: {e}")],
                        retry_count: 0,
                    },
                )
            }
        };

    // ─── Build Response ──────────────────────────────────
    let sources: Vec<SourceDocument> = evidence
        .iter()
        .map(|doc| SourceDocument {
            text: doc.text.clone(),
            doc_title: doc.doc_title.clone(),
            relevance_score: doc.score,
            level: doc.level,
        })
        .collect();

    let processing_time = start.elapsed().as_millis();
    info!("✅ Query processed in {}ms", processing_time);

    HttpResponse::Ok().json(QueryResponse {
        answer: final_answer,
        sources,
        agent_trace: AgentTrace {
            router_decision: "LEGAL".to_string(),
            hyde_document: expanded_queries, // Đổi tên logic ở mức payload JSON nhưng tái sử dụng field
            retrieved_count: evidence.len(),
            reranked_count: evidence.len(),
            analyst_reasoning: analyst_answer,
            compliance_check: compliance_result,
        },
        processing_time_ms: processing_time,
    })
}

/// Xây dựng câu hỏi có ngữ cảnh từ lịch sử hội thoại
fn build_contextualized_question(
    question: &str,
    history: &[ChatHistoryMessage],
) -> String {
    if history.is_empty() {
        return question.to_string();
    }

    // Giữ tối đa 3 cặp hội thoại gần nhất (6 messages)
    let recent: Vec<&ChatHistoryMessage> = history.iter().rev().take(6).collect::<Vec<_>>().into_iter().rev().collect();

    let mut context = String::from("Lịch sử hội thoại gần đây:\n");
    for msg in &recent {
        let role_label = if msg.role == "user" { "Người hỏi" } else { "Trợ lý" };
        let content_preview = &msg.content[..msg.content.floor_char_boundary(200)];
        context.push_str(&format!("- {}: {}\n", role_label, content_preview));
    }
    context.push_str(&format!("\nCâu hỏi hiện tại: {}", question));
    context
}
