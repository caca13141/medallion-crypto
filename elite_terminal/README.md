# 👽 TOPOOMEGA WAR ROOM DASHBOARD

**The Alien-Tech Native Interface for High-Frequency Topology Trading**

## Architecture
- **Core:** Tauri v2 (Rust + Webview)
- **Backend:** Rust (`src-tauri/src/main.rs`) - Handles WebSocket telemetry & System Tray
- **Frontend:** React 18 + TypeScript + Vite
- **Visuals:** Three.js (R3F) + TailwindCSS + Framer Motion
- **Performance:** Native binary, <200MB RAM, 144FPS

## Features
- **Real-time Topology Cloud:** 3D visualization of persistence homology (H1 cycles)
- **Live Equity Curve:** Animated Recharts area chart
- **TTI Gauge:** Topological Turbulence Index monitoring
- **Cyberpunk UI:** Dark mode, neon glows, glassmorphism
- **Zero-Copy Telemetry:** Direct Rust-to-Frontend event emission

## How to Launch

### Prerequisites
1. **Rust:** `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
2. **Node.js:** v18+

### Quick Start
Run the launch script:
```bash
./launch_alien_dashboard.sh
```

### Manual Start
```bash
cd alien_dashboard
npm install
npm run tauri dev
```

## Telemetry Integration
The dashboard listens on `ws://127.0.0.1:9001` for JSON telemetry from the Python engine.
Ensure your Python engine is pushing updates to this port.
