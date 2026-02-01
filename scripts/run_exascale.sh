#!/bin/bash
# Exascale 13.6B Launch Script
# 2026 Production Standard

export PYTHONPATH=$PYTHONPATH:$(pwd)
export MASTER_ADDR=localhost
export MASTER_PORT=12355

echo "🚀 INITIATING EXASCALE 13.6B SCALING..."

# Run the 13.6B training pass
# Using torchrun for distributed initialization
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 src/training/exascale_trainer.py fine

echo "✅ EXASCALE LAUNCH COMMAND ISSUED."
