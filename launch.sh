#!/bin/bash

# MEDALLION LAUNCH SCRIPT
# Starts the Engine (Background) and Dashboard (Foreground)

# 1. Activate Virtual Environment
source .venv/bin/activate

# 2. Start Engine in Background
echo "🦅 STARTING MEDALLION ENGINE..."
nohup python main.py > engine.log 2>&1 &
ENGINE_PID=$!
echo "   ENGINE PID: $ENGINE_PID"

# 3. Start Dashboard
echo "📊 STARTING WAR ROOM DASHBOARD..."
streamlit run dashboard.py

# 4. Cleanup on Exit
cleanup() {
    echo "🛑 SHUTTING DOWN..."
    kill $ENGINE_PID
    exit
}

trap cleanup SIGINT SIGTERM

wait
