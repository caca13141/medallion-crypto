#!/bin/bash
# Master Training Script - Trains all models sequentially
# Run this after data collection is complete

set -e  # Exit on error

echo "========================================"
echo "PHASE 16: COMPREHENSIVE MODEL TRAINING"
echo "========================================"
echo ""

# Check if data exists
if [ ! -f "data/raw/btc_ohlcv_2023_2025.parquet" ]; then
    echo "⚠️  Data not found. Running data collector first..."
    python3 src/training/data_collector.py
fi

echo ""
echo "========================================"
echo "STEP 1: Regime Detector (HMM)"
echo "========================================"
python3 src/training/train_regime_detector.py

echo ""
echo "========================================"
echo "STEP 2: Hawkes Process"
echo "========================================"
python3 src/training/train_hawkes.py

echo ""
echo "========================================"
echo "STEP 3: Topology Forecaster (GPU Required)"
echo "========================================"
echo "⚠️  This requires GPU. Skip if not available."
read -p "Train Topology Forecaster? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 src/training/train_topology_forecaster.py
else
    echo "⏭️  Skipped Topology Forecaster"
fi

echo ""
echo "========================================"
echo "STEP 4: NeuralCDE (GPU Required)"
echo "========================================"
echo "⚠️  This requires GPU. Skip if not available."
read -p "Train NeuralCDE? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 src/training/train_neural_cde.py
else
    echo "⏭️  Skipped NeuralCDE"
fi

echo ""
echo "========================================"
echo "STEP 5: PPO Agent (GPU Required)"
echo "========================================"
echo "⚠️  This requires GPU and takes 48+ hours. Skip if not available."
read -p "Train PPO Agent? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 src/training/train_ppo_agent.py --total_timesteps 1000000
else
    echo "⏭️  Skipped PPO Agent"
fi

echo ""
echo "========================================"
echo "✅ TRAINING COMPLETE"
echo "========================================"
echo ""
echo "Models saved to:"
ls -lh src/data/models/
echo ""
echo "Next: Run feed_dashboard.py to use trained models in production"
