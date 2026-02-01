#!/bin/bash
# Elite Terminal Institutional Launch Script
# Optimized for Marksman Cadence v4.2

echo "--------------------------------------------------"
echo " INITIALIZING ELITE TERMINAL PROTOCOL"
echo " STATUS: PRISTINE"
echo "--------------------------------------------------"

# 1. Clean up existing processes
echo " Cleaning existing telemetry streams..."
pkill -f feed_dashboard.py
pkill -f cargo

# 2. Launch Dashboard Server (Rust)
echo " Launching Institutional Feed Server..."
cd dashboard_server && cargo run --release > /dev/null 2>&1 &
sleep 2

# 3. Launch Marksman Feeder (Python)
echo " Initializing Marksman Data Pulse (2.0s Cadence)..."
cd .. && python3 feed_dashboard.py > /dev/null 2>&1 &

# 4. Launch Elite Terminal Frontend
echo " Starting Elite Terminal v4.2 Interface..."
cd elite_terminal && npm run dev
