import numpy as np
import torch
import time
from typing import Tuple, List
from src.validation.monte_carlo import MonteCarloEngine
from src.topology.engine import OrderBookTopology
from src.topology.utils import persistence_diagram_to_image
import os
from tqdm import tqdm

class SyntheticManifoldGenerator:
    """
    Generates millions of synthetic L3 snapshots for Exascale pre-training.
    Bridges Stochastic Differential Equations with Topological Persistence.
    """
    def __init__(self, resolution=32):
        self.mc = MonteCarloEngine(n_simulations=1) # We use it as an SDE stepper
        self.topo = OrderBookTopology()
        self.resolution = resolution
        
    def generate_synthetic_l3(self, price: float, volatility: float) -> dict:
        """
        Creates a synthetic L3 book based on a price point.
        Distributes liquidity using a power-law around the mid.
        """
        n_levels = 50
        # Spread follows volatility
        spread = price * volatility * 0.001
        
        bid_prices = price - spread - np.arange(n_levels) * (spread * 0.5)
        ask_prices = price + spread + np.arange(n_levels) * (spread * 0.5)
        
        # Sizes follow power law (Pareto distribution)
        bid_sizes = np.random.pareto(1.5, n_levels) + 0.1
        ask_sizes = np.random.pareto(1.5, n_levels) + 0.1
        
        return {
            "bids": np.column_stack([bid_prices, bid_sizes]),
            "asks": np.column_stack([ask_prices, ask_sizes])
        }

    def generate_sequence(self, length: int = 72) -> Tuple[torch.Tensor, float]:
        """
        Generates a sequence of 72 persistence images.
        """
        base_price = 90000.0
        vol = 0.02
        
        # 1. Generate Price Path via SDE
        # We use a single path from our MC engine
        path = self.mc.generate_price_paths(base_price, 0.0, vol, horizon=length)[0]
        
        images = []
        for p in path:
            # 2. Generate Synthetic L3
            l3 = self.generate_synthetic_l3(p, vol)
            
            # 3. Compute Persistence Image
            # Using the engine's internal logic for consistency
            from src.topology.engine import gudhi
            if gudhi:
                bids_np = l3['bids']
                asks_np = l3['asks']
                mid = (bids_np[0, 0] + asks_np[0, 0]) / 2.0
                # Feature cloud: [Price-Mid, Size]
                pts = np.vstack([
                    np.column_stack([bids_np[:, 0] - mid, bids_np[:, 1]]),
                    np.column_stack([asks_np[:, 0] - mid, asks_np[:, 1]])
                ])
                rips = gudhi.RipsComplex(points=pts, max_edge_length=100.0)
                st = rips.create_simplex_tree(max_dimension=2)
                st.persistence()
                diag = st.persistence_intervals_in_dimension(1)
                img = persistence_diagram_to_image(diag, resolution=self.resolution)
                images.append(img)
            else:
                images.append(np.zeros((1, self.resolution, self.resolution)))
                
        # Target: Return at end of 24 step horizon
        # (Simplified for pre-training)
        label = (path[-1] / path[length//2]) - 1.0
        
        return torch.from_numpy(np.array(images)).float(), float(label)

def run_generation(num_samples=1000):
    gen = SyntheticManifoldGenerator()
    os.makedirs("src/data/topology_dataset/synthetic", exist_ok=True)
    
    print(f"Generating {num_samples} Exascale pre-training samples...")
    for i in tqdm(range(num_samples)):
        seq, label = gen.generate_sequence()
        # Save as individual tensors for memory-mapped loading
        torch.save((seq, label), f"src/data/topology_dataset/synthetic/sample_{i}.pt")

if __name__ == "__main__":
    run_generation(num_samples=100) # Fast check
