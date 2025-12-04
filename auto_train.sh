#!/bin/bash

# AUTO-TRAIN: Waits for topology dataset then trains Transformer
# Run this in background to automatically start training when data is ready

echo "🦅 TOPOOMEGA AUTO-TRAINER"
echo "Waiting for topology dataset to complete..."

# Wait for dataset files to exist
while [ ! -f "src/data/topology_dataset/train.pkl" ]; do
    echo "⏳ $(date '+%H:%M:%S') - Still waiting for dataset..."
    sleep 60  # Check every minute
done

echo "✅ Dataset ready! Starting Transformer training..."

# Activate venv and train
source .venv/bin/activate
python src/training/train_transformer.py

echo "✅ Auto-training complete!"
