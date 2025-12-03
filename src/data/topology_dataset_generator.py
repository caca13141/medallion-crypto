"""
Topology Dataset Generator
Computes persistence features from historical data and creates train/val/test splits
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from src.topology.integrator import TopologyIntegrator

class TopologyDatasetGenerator:
    """Generate topology training dataset from historical candles"""
    
    def __init__(self, lookback=100, forecast_horizon=48):
        self.topology = TopologyIntegrator(lookback=lookback, resolution=20)
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon  # 48 candles = 12 hours (15m intervals)
        
    def generate_dataset(self, parquet_path):
        """
        Generate topology dataset from parquet file.
        
        Returns:
            dict with:
            - persistence_images_seq: (n_samples, 72, 20, 20) - for Transformer
            - labels_time: (n_samples,) - time until H1 dissolution (0-47)
            - labels_strength: (n_samples,) - dissolution strength [0, 1]
            - metadata: loop_score, tti, etc.
        """
        print(f"Loading {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        print(f"Total candles: {len(df)}")
        print(f"Date range: {df['t'].min()} to {df['t'].max()}")
        
        # Drop timestamp column for topology analysis
        df_features = df[['c', 'h', 'l', 'v']].copy()
        
        dataset = {
            'persistence_images_h1': [],
            'loop_scores': [],
            'ttis': [],
            'labels_time': [],
            'labels_strength': [],
            'timestamps': []
        }
        
        # Sliding window
        window_size = self.lookback
        seq_length = 72  # Need 72 images for Transformer

        print(f"\nGenerating topology features...")
        print(f"Window size: {window_size}, Sequence length: {seq_length}")
        
        for i in tqdm(range(seq_length + window_size, len(df_features) - self.forecast_horizon)):
            try:
                # Get window for current timestep
                window = df_features.iloc[i-window_size:i]
                
                # Compute topology
                topo_result = self.topology.analyze(window)
                
                # Store metadata
                dataset['loop_scores'].append(topo_result['loop_score'])
                dataset['ttis'].append(topo_result['tti'])
                dataset['persistence_images_h1'].append(topo_result['persistence_image_h1'])
                dataset['timestamps'].append(df.iloc[i]['t'])
                
                # Compute label: find next H1 dissolution
                # We'll compute topology for next 48 candles and find when loop score drops
                current_loop = topo_result['loop_score']
                
                dissolution_time = -1  # No dissolution found
                dissolution_strength = 0.0
                
                for j in range(1, self.forecast_horizon + 1):
                    future_window = df_features.iloc[i+j-window_size:i+j]
                    future_topo = self.topology.analyze(future_window)
                    future_loop = future_topo['loop_score']
                    
                    # Check if loop dissolved (significant drop)
                    if current_loop > 0 and future_loop < 0.5 * current_loop:
                        dissolution_time = j
                        dissolution_strength = (current_loop - future_loop) / (current_loop + 1e-8)
                        break
                
                # If no dissolution, label as max time + low strength
                if dissolution_time == -1:
                    dissolution_time = self.forecast_horizon - 1
                    dissolution_strength = 0.1
                    
                dataset['labels_time'].append(dissolution_time)
                dataset['labels_strength'].append(min(1.0, dissolution_strength))
                
            except Exception as e:
                # Skip on error
                continue
        
        # Convert to arrays
        for key in dataset:
            if key != 'timestamps':
                dataset[key] = np.array(dataset[key])
        
        print(f"\n✅ Generated {len(dataset['loop_scores'])} samples")
        print(f"Loop score range: [{dataset['loop_scores'].min():.3f}, {dataset['loop_scores'].max():.3f}]")
        print(f"TTI range: [{dataset['ttis'].min():.3f}, {dataset['ttis'].max():.3f}]")
        print(f"Dissolution time range: [0, {dataset['labels_time'].max()}]")
        
        return dataset
    
    def create_train_val_test_splits(self, dataset, train_ratio=0.6, val_ratio=0.2):
        """Split dataset chronologically (walk-forward)"""
        n = len(dataset['loop_scores'])
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train = {k: v[:train_end] if isinstance(v, np.ndarray) else v[:train_end] for k, v in dataset.items()}
        val = {k: v[train_end:val_end] if isinstance(v, np.ndarray) else v[train_end:val_end] for k, v in dataset.items()}
        test = {k: v[val_end:] if isinstance(v, np.ndarray) else v[val_end:] for k, v in dataset.items()}
        
        print(f"\nSplit sizes:")
        print(f"  Train: {len(train['loop_scores'])}")
        print(f"  Val: {len(val['loop_scores'])}")
        print(f"  Test: {len(test['loop_scores'])}")
        
        return train, val, test
    
    def save_datasets(self, train, val, test):
        """Save to pickle"""
        os.makedirs('src/data/topology_dataset', exist_ok=True)
        
        with open('src/data/topology_dataset/train.pkl', 'wb') as f:
            pickle.dump(train, f)
            
        with open('src/data/topology_dataset/val.pkl', 'wb') as f:
            pickle.dump(val, f)
            
        with open('src/data/topology_dataset/test.pkl', 'wb') as f:
            pickle.dump(test, f)
            
        print("\n💾 Saved datasets to src/data/topology_dataset/")

if __name__ == "__main__":
    generator = TopologyDatasetGenerator(lookback=100, forecast_horizon=48)
    
    # Generate from BTC data
    dataset = generator.generate_dataset('src/data/historical/btc_15m.parquet')
    
    # Split
    train, val, test = generator.create_train_val_test_splits(dataset)
    
    # Save
    generator.save_datasets(train, val, test)
    
    print("\n✅ Phase 1 Complete: Topology dataset ready for training!")
