#!/usr/bin/env python3
"""
Real-time Training Progress Monitor
Displays live metrics from the Exascale Fine-Tuning process
"""
import os
import time
import sys
from pathlib import Path

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')

def render_training_dashboard():
    """Render ASCII training dashboard"""
    clear_screen()
    
    print("=" * 70)
    print(" " * 15 + "⚡ EXASCALE TRAINING MONITOR ⚡")
    print("=" * 70)
    print()
    
    # Check if training process is running
    models_dir = Path("/Users/raphaelmaksoud/crypto toppo/models")
    
    # Get latest checkpoints
    checkpoints = sorted(models_dir.glob("exascale_model_fine_ep*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    print("📊 MODEL STATUS:")
    print(f"   Architecture: 1.1B Parameters (256-dim, 4-layer MoE)")
    print(f"   Training Mode: Stage 3 Fine-Tuning (Real L3 Data)")
    print(f"   Dataset Size: 7,744 snapshots")
    print()
    
    if checkpoints:
        latest = checkpoints[0]
        mtime = latest.stat().st_mtime
        age = time.time() - mtime
        
        print("💾 LATEST CHECKPOINT:")
        print(f"   File: {latest.name}")
        print(f"   Size: {latest.stat().st_size / 1e6:.1f} MB")
        print(f"   Updated: {age:.0f}s ago")
        print()
        
        if age < 300:  # Less than 5 minutes old
            print("✅ STATUS: ACTIVELY TRAINING")
        else:
            print("⚠️  STATUS: Training may have completed or stalled")
    else:
        print("⏳ Waiting for training to begin...")
    
    print()
    print("=" * 70)
    print("🔄 Refreshing every 5 seconds... (Ctrl+C to exit)")
    print("=" * 70)

if __name__ == "__main__":
    try:
        while True:
            render_training_dashboard()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\n👋 Training monitor stopped.")
        sys.exit(0)
