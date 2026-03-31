//! Query endpoint - 3-Agent Cross-lingual Pipeline
//! Agent 1: RAG-Router (classify + translate + expand + retrieve)
//! Agent 2: Analyst + Self-check (generate VN answer from EN context)
//! Agent 3: Reviewer (conditional fact-checking)

use actix_web::{post, web, HttpResponse};
use std::time::Instant;
use tracing::info;

use crate::agents::analyst::AnalystAgent;
use crate::agents::compliance::ReviewerAgent;
use crate::agents::rag::{QueryIntent, RagRouterAgent};
use crate::models::query::{
    AgentTrace, ChatHistoryMessage, ReviewerResult, QueryRequest, QueryResponse, SourceDocument,
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

    let contextualized_question = build_contextualized_question(question, history);

    // ═══════════════════════════════════════════════════════════════
    // AGENT 1: RAG-Router
    // Step 1a: Classify intent (parallel)
    // Step 1b: Translate VN → EN (parallel)
    // ═══════════════════════════════════════════════════════════════
    let intent_future = RagRouterAgent::classify(&groq, &contextualized_question);
    let translate_future = RagRouterAgent::translate_to_english(&groq, &contextualized_question);

    let (intent_res, translate_res) = tokio::join!(intent_future, translate_future);

    let intent = match intent_res {
        Ok(i) => i,
        Err(e) => {
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("Router error: {e}")
            }));
        }
    };

    let translated_query = match translate_res {
        Ok(t) => t,
        Err(e) => {
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("Translation error: {e}")
            }));
        }
    };

    // Casual → trả lời trực tiếp (không cần RAG)
    if intent == QueryIntent::Casual {
        let casual_answer = RagRouterAgent::casual_response(&groq, &contextualized_question)
            .await
            .unwrap_or_else(|_| "Xin lỗi, tôi không thể xử lý yêu cầu này.".to_string());

        return HttpResponse::Ok().json(QueryResponse {
            answer: casual_answer,
            sources: vec![],
            agent_trace: AgentTrace {
                router_decision: "CASUAL".to_string(),
                translated_query: String::new(),
                expanded_queries: String::new(),
                retrieved_count: 0,
                reranked_count: 0,
                analyst_answer: String::new(),
                reviewer_triggered: false,
                reviewer_result: ReviewerResult {
                    is_approved: true,
                    issues: vec![],
                    retry_count: 0,
                },
            },
            processing_time_ms: start.elapsed().as_millis(),
        });
    }

    // Step 1c: Multi-Query Expansion (EN)
    let expanded_queries = match RagRouterAgent::expand_queries_en(&groq, &translated_query).await {
        Ok(eq) => eq,
        Err(e) => {
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("Multi-Query Expansion error: {e}")
            }));
        }
    };

    // Step 1d: Hybrid Search + Rerank
    let evidence = match RagRouterAgent::retrieve(
        &groq,
        &qdrant,
        &translated_query,
        &expanded_queries,
        top_k
    ).await {
        Ok(result) => result,
        Err(e) => {
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("Retrieval error: {e}")
            }));
        }
    };

    // ═══════════════════════════════════════════════════════════════
    // AGENT 2: Analyst + Self-check
    // ═══════════════════════════════════════════════════════════════
    let analyst_answer = match AnalystAgent::analyze(&groq, &contextualized_question, &evidence).await {
        Ok(answer) => answer,
        Err(e) => {
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("Analyst error: {e}")
            }));
        }
    };

    // ═══════════════════════════════════════════════════════════════
    // AGENT 3: Conditional Reviewer
    // Chỉ chạy khi query hỏi numbers/metrics/formulas
    // ═══════════════════════════════════════════════════════════════
    let reviewer_needed = AnalystAgent::needs_review(&contextualized_question)
        || AnalystAgent::needs_review(&translated_query);

    let (final_answer, reviewer_result, reviewer_triggered) = if reviewer_needed {
        info!("🔍 Reviewer triggered for this query");
        match ReviewerAgent::check_with_retry(&groq, question, &analyst_answer, &evidence).await {
            Ok((answer, result)) => (answer, result, true),
            Err(e) => {
                // Reviewer fail → dùng answer gốc
                (
                    analyst_answer.clone(),
                    ReviewerResult {
                        is_approved: false,
                        issues: vec![format!("Reviewer error: {e}")],
                        retry_count: 0,
                    },
                    true,
                )
            }
        }
    } else {
        info!("⚡ Reviewer skipped (not needed)");
        (
            analyst_answer.clone(),
            ReviewerResult {
                is_approved: true,
                issues: vec![],
                retry_count: 0,
            },
            false,
        )
    };

    // ═══════════════════════════════════════════════════════════════
    // Build Response
    // ═══════════════════════════════════════════════════════════════
    let sources: Vec<SourceDocument> = evidence
        .iter()
        .map(|doc| SourceDocument {
            text: doc.text.clone(),
            doc_title: doc.doc_title.clone(),
            authors: doc.authors.clone(),
            year: doc.year,
            arxiv_id: doc.arxiv_id.clone(),
            relevance_score: doc.score,
            level: doc.level,
        })
        .collect();

    let processing_time = start.elapsed().as_millis();
    info!("✅ Query processed in {}ms (reviewer: {})", processing_time, reviewer_triggered);

    HttpResponse::Ok().json(QueryResponse {
        answer: final_answer,
        sources,
        agent_trace: AgentTrace {
            router_decision: "TECHNICAL".to_string(),
            translated_query,
            expanded_queries,
            retrieved_count: evidence.len(),
            reranked_count: evidence.len(),
            analyst_answer,
            reviewer_triggered,
            reviewer_result,
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
