use axum::{
    extract::{ws::{Message, WebSocket, WebSocketUpgrade}, State},
    response::IntoResponse,
    routing::get,
    Router,
};
use futures::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, sync::{Arc, Mutex}};
use tokio::sync::broadcast;
use tower_http::cors::CorsLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod engine;
use engine::{spectral::SpectralEngine, attractor::AttractorEngine, ghost::GhostEngine};

// Telemetry Data Structure
#[derive(Clone, Debug, Serialize, Deserialize)]
struct TelemetryUpdate {
    topic: String, // "l3_book", "tti", "signals", "positions", "risk", "trades", "raw_trade", "raw_l3"
    payload: serde_json::Value,
    timestamp: u64,
}

// App State
struct AppState {
    tx: broadcast::Sender<TelemetryUpdate>,
    spectral: Arc<Mutex<SpectralEngine>>,
    attractor: Arc<Mutex<AttractorEngine>>,
    ghost: Arc<Mutex<GhostEngine>>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let (tx, _rx) = broadcast::channel(100);
    
    // Initialize Engines
    let spectral = Arc::new(Mutex::new(SpectralEngine::new(1024))); // 1024 tick window
    let attractor = Arc::new(Mutex::new(AttractorEngine::new(1000, 5, 3))); // 1000 history, tau=5, dim=3
    let ghost = Arc::new(Mutex::new(GhostEngine::new()));

    let app_state = Arc::new(AppState { 
        tx,
        spectral,
        attractor,
        ghost
    });

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/push", axum::routing::post(push_handler)) // Endpoint for Python to push updates
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    tracing::info!("Dashboard Server listening on 0.0.0.0:3000");
    axum::serve(listener, app).await.unwrap();
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    tracing::info!("WebSocket client connected!");
    let (mut sender, mut receiver) = socket.split();
    let mut rx = state.tx.subscribe();
    tracing::info!("Subscribed to broadcast channel, receiver count: {}", state.tx.receiver_count());

    // Spawn a task to forward broadcast messages to this websocket
    let mut send_task = tokio::spawn(async move {
        tracing::info!("Send task started, waiting for broadcasts...");
        loop {
            match rx.recv().await {
                Ok(msg) => {
                    tracing::info!("Received broadcast: {}", msg.topic);
                    // Filter out internal "raw" messages to save bandwidth
                    if msg.topic.starts_with("raw_") {
                        continue;
                    }
                    if let Ok(json) = serde_json::to_string(&msg) {
                        tracing::info!("Sending to WebSocket client: {} bytes", json.len());
                        if sender.send(Message::Text(json)).await.is_err() {
                            tracing::warn!("WebSocket send failed, closing");
                            break;
                        }
                    }
                }
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    tracing::warn!("Client lagged, skipped {} messages", skipped);
                    continue;
                }
                Err(broadcast::error::RecvError::Closed) => {
                    tracing::info!("Broadcast channel closed");
                    break;
                }
            }
        }
    });

    // Keep connection alive until client disconnects
    while let Some(Ok(_)) = receiver.next().await {
        // We don't expect messages from client, but keep loop open
    }
    tracing::info!("WebSocket client disconnected");

    send_task.abort();
}

// Endpoint for Python engine to push updates
async fn push_handler(
    State(state): State<Arc<AppState>>,
    axum::Json(payload): axum::Json<TelemetryUpdate>,
) -> impl IntoResponse {
    tracing::info!("📡 Telemetry Push: {}", payload.topic);
    let _ = state.tx.send(payload.clone());

    // 2. Process Raw Data in Rust Engines
    if payload.topic == "raw_trade" {
        if let Some(trade) = payload.payload.as_object() {
             if let Some(timestamp) = trade.get("timestamp").and_then(|t| t.as_f64()) {
                 // Calculate inter-arrival time (simplified, assumes ordered)
                 // In a real system we'd track last timestamp per symbol
                 // Here we just use a dummy delta or the timestamp itself if it's relative
                 // Let's assume payload has "latency_ms" or similar we can use as a proxy for now
                 // Or better, just push the price to Attractor
                 
                 if let Some(price) = trade.get("price").and_then(|p| p.as_f64()) {
                     let mut attractor = state.attractor.lock().unwrap();
                     // attractor.update(price as f32);
                     
                     // Broadcast Attractor Update (DISABLED for Marksman Cadence)
                     /*
                     let traj = attractor.get_trajectory();
                     let update = TelemetryUpdate {
                         topic: "phase_attractor".to_string(),
                         payload: serde_json::to_value(traj).unwrap(),
                         timestamp: payload.timestamp
                     };
                     let _ = state.tx.send(update);
                     */
                 }
                 
                 // Spectral Update
                 if let Some(latency) = trade.get("latency_ms").and_then(|l| l.as_f64()) {
                     // let mut spectral = state.spectral.lock().unwrap();
                     // spectral.update(latency as f32);
                     
                     /*
                     let spectrum = spectral.compute_spectrum();
                     let spectrum_json: Vec<serde_json::Value> = spectrum.iter().map(|(f, p)| {
                         serde_json::json!({ "freq": f, "power": p })
                     }).collect();
                     
                     let update = TelemetryUpdate {
                         topic: "spectral_resonance".to_string(),
                         payload: serde_json::Value::Array(spectrum_json),
                         timestamp: payload.timestamp
                     };
                     let _ = state.tx.send(update);
                     */
                 }
             }
        }
    } else if payload.topic == "raw_l3" {
        // Process L3 for Ghost Liquidity
        if let Some(book) = payload.payload.as_object() {
             // Extract bids/asks
             // This parsing is verbose in Rust with serde_json::Value, simplified for brevity
             // In prod we use strong types
             
             // Mocking the extraction for now to show flow
             let mut ghost = state.ghost.lock().unwrap();
             // ghost.detect(...)
             // Broadcast "ghost_liquidity"
        }
    }

    axum::http::StatusCode::OK
}
