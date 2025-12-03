#!/bin/bash

# TOPOOMEGA v2.0 LAUNCH SCRIPT
# Nuclear persistent homology warfare

# 1. Activate Virtual Environment
source .venv/bin/activate

echo "=" >> "="
echo "🦅 TOPOOMEGA v2.0 - NUCLEAR CRYPTO WARFARE"
echo "Persistent Homology | Bifiltrated Persistence | Wasserstein-PPO"
echo "Target: 450-650% CAGR | Sharpe 15-19 | MaxDD <7%"
echo "="*60

# 2. Start Engine in Background
echo "⚙️  STARTING TOPOOMEGA ENGINE..."
nohup python topo_omega_main.py > topo_engine.log 2>&1 &
ENGINE_PID=$!
echo "   ENGINE PID: $ENGINE_PID"

# 3. Start Dashboard
echo "📊 STARTING TOPOLOGY WAR ROOM..."
streamlit run topo_dashboard.py

# 4. Cleanup on Exit
cleanup() {
    echo "🛑 SHUTTING DOWN TOPOOMEGA..."
    kill $ENGINE_PID
    exit
}

trap cleanup SIGINT SIGTERM

wait
