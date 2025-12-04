"""
PRODUCTION TOPOLOGY GENERATOR
Generates 32x32 Persistence Images + H1 Summaries for 207k candles.
Uses GUDHI + Ripser++ (Production Engine).
"""

import sys
import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from ripser import ripser
from persim import PersistenceImager
import warnings

# Suppress TDA warnings
warnings.filterwarnings("ignore")

class ProductionTopologyGenerator:
    def __init__(self, resolution=32, lookback=50):
        self.resolution = resolution
        self.lookback = lookback
        
        # Initialize Imager
        # Resolution is determined by birth_range and pixel_size
        # For 32x32 over range (0, 2.0), pixel_size should be 2.0/32 = 0.0625
        pixel_size = 2.0 / resolution
        self.imager = PersistenceImager(pixel_size=pixel_size, birth_range=(0, 2.0))
        
    def compute_window_topology(self, window_prices):
        """Compute topology for a single window"""
        # Normalize window
        norm_prices = (window_prices - np.mean(window_prices)) / (np.std(window_prices) + 1e-8)
        
        # Time-delay embedding (Takens)
        # 2D embedding: (p[t], p[t-1])
        point_cloud = np.column_stack([norm_prices[:-1], norm_prices[1:]])
        
        # Compute Persistence
        try:
            diagrams = ripser(point_cloud, maxdim=1)['dgms']
            
            # H1 Persistence Image
            if len(diagrams) > 1 and len(diagrams[1]) > 0:
                h1 = diagrams[1]
                # Fix infinite death
                max_death = np.max(h1[h1[:, 1] != np.inf][:, 1]) if np.any(h1[:, 1] != np.inf) else 1.0
                h1_clean = np.copy(h1)
                h1_clean[h1_clean[:, 1] == np.inf, 1] = max_death * 1.1
                
                # Ensure shape is correct
                if p_image.shape != (self.resolution, self.resolution):
                    p_image = np.zeros((self.resolution, self.resolution))
                
                # H1 Summary (8-dim)
                lifetimes = h1_clean[:, 1] - h1_clean[:, 0]
                summary = np.array([
                    np.max(lifetimes),
                    np.mean(lifetimes),
                    np.sum(lifetimes),
                    np.std(lifetimes),
                    np.max(h1_clean[:, 0]), # Max Birth
                    np.max(h1_clean[:, 1]), # Max Death
                    len(lifetimes),
                    0.0 # Placeholder
                ])
            else:
                p_image = np.zeros((self.resolution, self.resolution))
                summary = np.zeros(8)
                
            return p_image, summary
            
        except Exception:
            return np.zeros((self.resolution, self.resolution)), np.zeros(8)

    def generate(self, parquet_path, output_path):
        print(f"🚀 Loading {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        prices = df.get('close', df.get('c')).values
        
        print(f"📊 Processing {len(prices):,} candles...")
        print(f"   Resolution: {self.resolution}x{self.resolution}")
        print(f"   Lookback: {self.lookback}")
        
        images = []
        summaries = []
        labels = []
        
        # Process in batches to save memory
        batch_size = 10000
        
        for i in tqdm(range(self.lookback, len(prices) - 24)):
            window = prices[i-self.lookback:i]
            
            # Compute Topology
            img, summ = self.compute_window_topology(window)
            
            # Label (Next 24h return)
            # 24h = 96 candles (15m)
            future_idx = min(i + 96, len(prices) - 1)
            ret = (prices[future_idx] / prices[i]) - 1
            
            # 3-class label
            if ret > 0.02: label = 2   # Long
            elif ret < -0.02: label = 0 # Short
            else: label = 1            # Neutral
            
            images.append(img)
            summaries.append(summ)
            labels.append(label)
            
            # Periodic save
            if len(images) >= batch_size:
                self._save_batch(images, summaries, labels, output_path, append=os.path.exists(output_path))
                images, summaries, labels = [], [], []
                
        # Save remaining
        if images:
            self._save_batch(images, summaries, labels, output_path, append=os.path.exists(output_path))
            
        print(f"✅ Generation Complete! Saved to {output_path}")

    def _save_batch(self, images, summaries, labels, path, append=False):
        data = {
            'images': np.array(images, dtype=np.float32),
            'summaries': np.array(summaries, dtype=np.float32),
            'labels': np.array(labels, dtype=np.int8)
        }
        
        mode = 'ab' if append else 'wb'
        with open(path, mode) as f:
            pickle.dump(data, f)

if __name__ == "__main__":
    gen = ProductionTopologyGenerator()
    gen.generate(
        'src/data/historical/btc_usdt_15m.parquet',
        'src/data/topology_dataset/production_topology.pkl'
    )
