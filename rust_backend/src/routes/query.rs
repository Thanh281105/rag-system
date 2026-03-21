//! Query endpoint - Luồng xử lý chính của hệ thống multi-agent

use actix_web::{post, web, HttpResponse};
use std::time::Instant;
use tracing::info;

use crate::agents::analyst::AnalystAgent;
use crate::agents::compliance::ComplianceAgent;
use crate::agents::rag::RagAgent;
use crate::agents::router::{QueryIntent, RouterAgent};
use crate::models::query::{
    AgentTrace, ComplianceResult, QueryRequest, QueryResponse, SourceDocument,
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

    info!("📨 Query: '{}'", &question[..question.len().min(80)]);

    // ─── Step 1: Router Agent ────────────────────────────
    let intent = match RouterAgent::classify(&groq, question).await {
        Ok(intent) => intent,
        Err(e) => {
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("Router error: {e}")
            }));
        }
    };

    // Casual → trả lời trực tiếp
    if intent == QueryIntent::Casual {
        let casual_answer = RouterAgent::casual_response(&groq, question)
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

    // ─── Step 2: RAG Agent ───────────────────────────────
    let (evidence, hyde_document) = match RagAgent::retrieve(&groq, &qdrant, question, top_k).await
    {
        Ok(result) => result,
        Err(e) => {
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("RAG error: {e}")
            }));
        }
    };

    // ─── Step 3: Analyst Agent ───────────────────────────
    let analyst_answer = match AnalystAgent::analyze(&groq, question, &evidence).await {
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
            hyde_document,
            retrieved_count: evidence.len(),
            reranked_count: evidence.len(),
            analyst_reasoning: analyst_answer,
            compliance_check: compliance_result,
        },
        processing_time_ms: processing_time,
    })
}
