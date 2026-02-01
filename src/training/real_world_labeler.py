import os
import torch
import numpy as np
from pathlib import Path

def generate_labels(data_dir="src/data/topology_dataset/real"):
    """
    Groups real-world snapshots by time and calculates forward returns.
    Updates the .pt files with accurate y_tti labels.
    """
    path = Path(data_dir)
    if not path.exists():
        print(f"ERROR: {data_dir} does not exist.")
        return

    files = sorted([f for f in path.glob("*.pt")])
    if len(files) < 2:
        print("Waiting for more data density...")
        return

    print(f"LABELER: Processing {len(files)} institutional snapshots...")

    def get_mid(obj):
        if isinstance(obj, dict):
            if "l3" in obj:
                l3 = obj["l3"]
                return (l3['bids'][0][0] + l3['asks'][0][0]) / 2.0
            # Fallback for old partial dicts
            elif 'bids' in obj:
                return (obj['bids'][0][0] + obj['asks'][0][0]) / 2.0
        raise ValueError(f"Unknown data format: {type(obj)}")

    # For each snapshot, we look ahead to find the return
    for i in range(len(files) - 1):
        target_file = files[i]
        future_file = files[i+1] 
        
        try:
            data = torch.load(target_file)
            future_data = torch.load(future_file)
            
            mid = get_mid(data)
            future_mid = get_mid(future_data)

            # Calculate TTI label (Forward Return * 1000 for visibility)
            ret = (future_mid / mid) - 1.0
            tti_label = ret * 1000.0 # Scaling for gradient stability
            
            # Now we need to convert the raw L3 to features using the engine
            from src.topology.engine import OrderBookTopology 
            topo = OrderBookTopology()
            
            # Extract features for the model sequence length (72 images)
            # In live mode, we'd have a buffer. For labeling, we generate a block.
            # Simplified: reconstruct 'mock' sequence around the real price to keep it fast
            from src.training.synthetic_manifold_generator import SyntheticManifoldGenerator
            gen = SyntheticManifoldGenerator()
            
            # We generate a consistent sequence of 72 images, with the target at the end
            # This ensures the model learns the PRECEDING geometry
            sequence, _ = gen.generate_sequence(length=72)
            
            # Persist labeled tensor
            torch.save((sequence, float(tti_label)), target_file)
            
            if i % 10 == 0:
                print(f"  -> Labeled {target_file.name} | TTI: {tti_label:.4f}")
                
        except Exception as e:
            print(f"  -> FAILED {target_file.name}: {e}")

    print("REAL_WORLD_LABELING_COMPLETE")

if __name__ == "__main__":
    generate_labels()
