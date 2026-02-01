#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::{Manager, Emitter};
use tokio::net::TcpListener;
use tokio_tungstenite::accept_async;
use futures_util::{StreamExt, SinkExt};
use std::sync::Arc;
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
struct WhaleTrade {
    price: f64,
    size: f64,
    value: f64,
    side: String,
    timestamp: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct FusionNarrative {
    narrative: String,
    status: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct OnChainMetrics {
    smart_money_score: f64,
    flow_persistence: f64,
    clusters: i32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct TelemetryData {
    timestamp: f64,
    pnl: f64,
    equity: f64,
    drawdown: f64,
    tti: f64,
    positions: Vec<Position>,
    topology: TopologySnapshot,
    #[serde(default)]
    onchain: Option<OnChainMetrics>,
    #[serde(default)]
    predictions: Vec<PricePoint>,
    #[serde(default)]
    actuals: Vec<PricePoint>,
    #[serde(default)]
    current_price: f64,
    #[serde(default)]
    whale_flow: Vec<WhaleTrade>,
    #[serde(default)]
    liquidations: Vec<serde_json::Value>,
    #[serde(default)]
    narrative: Option<FusionNarrative>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct PricePoint {
    timestamp: f64,
    price: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Position {
    symbol: String,
    side: String,
    size: f64,
    pnl: f64,
    leverage: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct TopologySnapshot {
    #[serde(default)]
    persistence_image: Vec<Vec<f64>>,
    betti_curves: Vec<Vec<f64>>,
    wasserstein_dist: f64,
    #[serde(default)]
    bull_loops: i32,
    #[serde(default)]
    bear_loops: i32,
    #[serde(default)]
    bifiltration_score: f64,
}

// Global state to hold the latest telemetry
struct AppState {
    latest_telemetry: Mutex<Option<TelemetryData>>,
}

// Tauri command to get latest telemetry (called from frontend)
#[tauri::command]
async fn get_latest_telemetry(state: tauri::State<'_, Arc<AppState>>) -> Result<Option<TelemetryData>, String> {
    let lock = state.latest_telemetry.lock().await;
    Ok(lock.clone())
}

#[tokio::main]
async fn main() {
    // Initialize logging
    env_logger::init();

    let app_state = Arc::new(AppState {
        latest_telemetry: Mutex::new(None),
    });

    tauri::Builder::default()
        .manage(app_state.clone())
        .invoke_handler(tauri::generate_handler![get_latest_telemetry])
        // .plugin(tauri_plugin_shell::init())
        .setup(move |app| {
            let app_handle = app.handle().clone();
            let state_clone = app_state.clone();

            // Spawn WebSocket Server to listen for Python Engine pushes
            tokio::spawn(async move {
                let addr = "127.0.0.1:9001";
                let listener = TcpListener::bind(&addr).await.expect("Failed to bind WS port");
                println!("🚀 Alien Dashboard Telemetry Link Active on {}", addr);

                while let Ok((stream, _)) = listener.accept().await {
                    let app_handle = app_handle.clone();
                    let state = state_clone.clone();
                    
                    tokio::spawn(async move {
                        let ws_stream = accept_async(stream).await.expect("Error during handshake");
                        let (mut _write, mut read) = ws_stream.split();

                        while let Some(msg) = read.next().await {
                            if let Ok(msg) = msg {
                                if msg.is_text() || msg.is_binary() {
                                    let text = msg.to_text().unwrap_or("{}");
                                    if let Ok(data) = serde_json::from_str::<TelemetryData>(text) {
                                        // Update state
                                        let mut lock = state.latest_telemetry.lock().await;
                                        *lock = Some(data.clone());
                                        
                                        // Emit to frontend (Zero-copy-ish via Tauri event)
                                        let _ = app_handle.emit("telemetry-update", data);
                                    }
                                }
                            }
                        }
                    });
                }
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
