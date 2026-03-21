//! Agentic RAG Legal System - Rust Backend
//! 
//! Multi-agent orchestration server với Actix-Web.
//! Agents: Router → RAG (HyDE + Search) → Analyst → Compliance

mod config;
mod errors;
mod models;
mod agents;
mod services;
mod routes;

use actix_cors::Cors;
use actix_files as fs;
use actix_web::{web, App, HttpServer, middleware};
use tracing::info;
use tracing_subscriber::EnvFilter;

use crate::config::AppConfig;
use crate::services::groq::GroqService;
use crate::services::qdrant::QdrantService;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // ─── Logging ─────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    // ─── Configuration ───────────────────────────────────
    let config = AppConfig::from_env().expect("Failed to load configuration");
    let bind_addr = format!("{}:{}", config.host, config.port);
    
    info!("🏛️ Legal RAG Backend starting...");
    info!("📍 Binding to: {}", bind_addr);

    // ─── Services ────────────────────────────────────────
    let groq_service = GroqService::new(&config);
    info!("✅ Groq service initialized");

    let qdrant_service = QdrantService::new(&config)
        .await
        .expect("Failed to connect to Qdrant");
    info!("✅ Qdrant service initialized");

    // ─── Server ──────────────────────────────────────────
    info!("🚀 Server ready at http://{}", bind_addr);

    HttpServer::new(move || {
        // CORS cho frontend
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600);

        App::new()
            // Middleware
            .wrap(cors)
            .wrap(middleware::Logger::default())
            .wrap(tracing_actix_web::TracingLogger::default())
            // Shared state
            .app_data(web::Data::new(groq_service.clone()))
            .app_data(web::Data::new(qdrant_service.clone()))
            // Routes
            .service(routes::health::health_check)
            .service(routes::query::handle_query)
            // Static files (frontend)
            .service(fs::Files::new("/", "../frontend").index_file("index.html"))
    })
    .bind(&bind_addr)?
    .run()
    .await
}
