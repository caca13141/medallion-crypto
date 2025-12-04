#!/bin/bash
# OVERNIGHT PRODUCTION RUN SCRIPT
# 1. Generate Topology
# 2. Train Transformer
# 3. Validate

echo "🌙 STARTING OVERNIGHT PRODUCTION RUN"
date

# 1. Generate Topology
echo "------------------------------------------------"
echo "STEP 1: Generating 207k Persistence Images..."
echo "------------------------------------------------"
"/Users/raphaelmaksoud/crypto toppo/.venv/bin/python" src/data/generate_production_topology.py

if [ $? -ne 0 ]; then
    echo "❌ Topology generation failed!"
    exit 1
fi

# 2. Train Transformer
echo "------------------------------------------------"
echo "STEP 2: Training 36-Layer Transformer..."
echo "------------------------------------------------"
"/Users/raphaelmaksoud/crypto toppo/.venv/bin/python" src/training/train_production_transformer.py

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

echo "------------------------------------------------"
echo "✅ OVERNIGHT RUN COMPLETE"
echo "------------------------------------------------"
date
