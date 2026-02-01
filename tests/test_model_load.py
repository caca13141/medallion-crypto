import asyncio
import json
import numpy as np
import pandas as pd
import ccxt
import torch
import websockets
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from collections import deque
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# NEW: Nuclear Engines
from src.signals.hawkes_cascade import process_tick as hawkes_process_tick
from src.signals.rough_path_signature import RoughPathEngine
from src.topology.persistence_core import ProductionTopologyEngine
from src.forecasting.topology_forecaster import create_model

# Initialize outside asyncio
print("Initializing Topology Engine (Global)...")
topo_engine = ProductionTopologyEngine(resolution=32)
print("Topology Engine Initialized")

async def main():
    print("Testing Model Load inside Asyncio...")
    
    model_path = 'models/transformer_best.pth'

    if not os.path.exists(model_path):
        print(f" File not found: {model_path}")
        return

    try:
        print("Creating model...")
        model = create_model()
        print("Model created.")
        
        print(f"Loading {model_path}...")
        state_dict = torch.load(model_path, map_location='cpu')
        print("State dict loaded.")
        
        model.load_state_dict(state_dict)
        model.eval()
        print(" Model loaded successfully!")
        print(f"Keys: {len(state_dict.keys())}")
    except Exception as e:
        print(f" Load failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
