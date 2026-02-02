#!/bin/bash
# Elite Terminal Complete System Launch via Ghostty
# Opens multiple terminal windows for comprehensive visualization

WORKSPACE="/Users/raphaelmaksoud/crypto toppo"
GHOSTTY="/Applications/Ghostty.app/Contents/MacOS/ghostty"

echo "=================================================="
echo " LAUNCHING ELITE TERMINAL IN GHOSTTY"
echo "=================================================="

# 1. Training Monitor (Main Window)
echo "[1/5] Opening Training Monitor..."
$GHOSTTY -e bash -c "cd '$WORKSPACE' && python3 scripts/training_monitor.py" &
sleep 1

# 2. Dashboard Server (Rust)
echo "[2/5] Launching Dashboard Server (Rust)..."
$GHOSTTY -e bash -c "cd '$WORKSPACE/dashboard_server' && echo '=== RUST DASHBOARD SERVER ===' && cargo run --release" &
sleep 3

# 3. Marksman Feeder (Python)
echo "[3/5] Launching Marksman Feeder (Python)..."
$GHOSTTY -e bash -c "cd '$WORKSPACE' && echo '=== MARKSMAN CADENCE FEEDER (2.0s) ===' && python3 feed_dashboard.py" &
sleep 2

# 4. Polyglot Execution Stack
echo "[4/5] Launching Polyglot Stack (C++/Rust/OCaml)..."
$GHOSTTY -e bash -c "cd '$WORKSPACE' && echo '=== POLYGLOT EXECUTION STACK ===' && bash scripts/launch_polyglot.sh" &
sleep 1

# 5. Frontend Dev Server
echo "[5/5] Launching Elite Terminal Frontend..."
$GHOSTTY -e bash -c "cd '$WORKSPACE/elite_terminal' && echo '=== ELITE TERMINAL FRONTEND (v4.2) ===' && npm run dev" &

echo ""
echo "=================================================="
echo "✅ ALL SYSTEMS ACTIVE IN GHOSTTY"
echo "=================================================="
echo ""
echo "Windows Launched:"
echo "  1. Training Monitor (Background)"
echo "  2. Rust Dashboard Server (Port 1420)"
echo "  3. Python Marksman Feeder"
echo "  4. Polyglot Stack (C++/Rust/OCaml)"
echo "  5. Frontend Dev Server (Port 3000)"
echo ""
echo "Access Elite Terminal at: http://localhost:3000"
echo ""
echo "Press Ctrl+C to keep this launcher alive."
echo "Close individual windows or use 'killall ghostty' to shutdown."
echo "=================================================="

# Keep launcher alive
wait
