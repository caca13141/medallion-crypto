#!/bin/bash
# Elite Terminal Optimized Launch Script
# "Maximum efficiency without garbage."

echo "🚀 Launching Elite Terminal (Stability Mode)..."

# 1. Cleanup
echo "-> Cleaning up old processes..."
pkill -9 -f "dashboard_server"
pkill -9 -f "feed_dashboard"
pkill -9 -f "vite"
lsof -ti :1420,3000 | xargs kill -9 2>/dev/null

# 2. Environment Setup
export RUST_LOG=info
export LIGHT_MODE=true # Force light mode for instant start

# 3. Start Rust Backend
echo "-> Starting Rust Backend (Port 3000)..."
cd dashboard_server
cargo build --release --quiet
nohup ./target/release/dashboard_server > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend
sleep 2

# 4. Start Data Feeder
echo "-> Starting Marksman Feeder..."
nohup /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -u feed_dashboard.py > feeder.log 2>&1 &
FEEDER_PID=$!

# 5. Start Frontend
echo "-> Starting Vite Frontend (Port 1420)..."
cd elite_terminal
nohup npm run dev -- --host > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo "✅ System Launched."
echo "   Backend PID: $BACKEND_PID"
echo "   Feeder PID: $FEEDER_PID"
echo "   Frontend PID: $FRONTEND_PID"
echo ""
echo "-> Tailing logs (Ctrl+C to exit logs, system keeps running)..."
echo "-------------------------------------------------------------"

# Tail logs
tail -f backend.log feeder.log frontend.log
