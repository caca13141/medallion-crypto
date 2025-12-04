"""
NUMBA-ACCELERATED Fast Topology Dataset
10-50x speedup with JIT compilation (no C++ needed)
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from numba import jit

@jit(nopython=True)
def compute_autocorr_fast(returns, lag):
    """Numba-accelerated autocorrelation"""
    n = len(returns)
    if n <= lag:
        return 0.0
    
    mean = np.mean(returns)
    var = np.var(returns)
    
    if var == 0:
        return 0.0
    
    autocorr = 0.0
    for i in range(n - lag):
        autocorr += (returns[i] - mean) * (returns[i + lag] - mean)
    
    return autocorr / ((n - lag) * var)

@jit(nopython=True)
def compute_features_fast(prices, volumes, highs, lows):
    """
    Numba-accelerated feature computation
    Returns 10D feature vector
    """
    n = len(prices)
    
    if n < 10:
        return np.zeros(10)
    
    # Log returns
    log_prices = np.log(prices)
    returns = np.diff(log_prices)
    
    # Volatility
    vol = np.std(returns)
    
    # Momentum (last 10 periods)
    momentum = np.mean(returns[-10:])
    
    # Autocorrelations (cycle detection)
    ac_1 = compute_autocorr_fast(returns, 1)
    ac_5 = compute_autocorr_fast(returns, 5)
    ac_10 = compute_autocorr_fast(returns, 10)
    
    # Volume change
    vol_change = np.mean(np.diff(volumes[-10:]))
    
    # High-Low range
    hl_range = np.mean((highs - lows) / prices)
    
    # Extremes
    max_up = np.max(returns)
    max_down = np.min(returns)
    avg_abs_return = np.mean(np.abs(returns))
    
    return np.array([
        vol,
        momentum,
        ac_1,
        ac_5,
        ac_10,
        vol_change,
        hl_range,
        max_up,
        max_down,
        avg_abs_return
    ])

class NumbaTopologyDataset:
    """Numba-accelerated topology dataset"""
    
    def __init__(self, lookback=50):
        self.lookback = lookback
        
    def generate(self, parquet_path):
        """Generate dataset with Numba acceleration"""
        print(f"Loading {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        # Convert to numpy for Numba
        prices = df['c'].values
        highs = df['h'].values
        lows = df['l'].values
        volumes = df['v'].values
        
        features_list = []
        labels_list = []
        
        print("Generating features (NUMBA-ACCELERATED)...")
        
        for i in tqdm(range(self.lookback, len(df) - 24)):
            # Window arrays
            window_prices = prices[i-self.lookback:i]
            window_highs = highs[i-self.lookback:i]
            window_lows = lows[i-self.lookback:i]
            window_volumes = volumes[i-self.lookback:i]
            
            # Compute features (JIT compiled - FAST!)
            features = compute_features_fast(
                window_prices, 
                window_volumes,
                window_highs,
                window_lows
            )
            
            # Label: next 6h return
            future_return = (prices[i+24] / prices[i]) - 1
            label = 1 if future_return > 0.01 else (-1 if future_return < -0.01 else 0)
            
            features_list.append(features)
            labels_list.append(label)
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"\n✅ Generated {len(X)} samples")
        print(f"Feature shape: {X.shape}")
        
        # Split
        n = len(X)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        
        dataset = {
            'X_train': X[:train_end],
            'y_train': y[:train_end],
            'X_val': X[train_end:val_end],
            'y_val': y[train_end:val_end],
            'X_test': X[val_end:],
            'y_test': y[val_end:]
        }
        
        # Save
        os.makedirs('src/data/topology_dataset', exist_ok=True)
        with open('src/data/topology_dataset/numba_dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
            
        print(f"\n💾 Saved to src/data/topology_dataset/numba_dataset.pkl")
        
        return dataset

if __name__ == "__main__":
    import time
    
    generator = NumbaTopologyDataset(lookback=50)
    
    print("🚀 NUMBA-ACCELERATED MODE")
    print("=" * 60)
    
    start = time.time()
    dataset = generator.generate('src/data/historical/btc_15m.parquet')
    elapsed = time.time() - start
    
    print(f"\n⚡ Total time: {elapsed:.2f}s")
    print(f"⚡ Speed: {len(dataset['X_train']) + len(dataset['X_val']) + len(dataset['X_test']):.0f} samples / {elapsed:.2f}s = {(len(dataset['X_train']) + len(dataset['X_val']) + len(dataset['X_test']))/elapsed:.0f} samples/sec")
    print("\n✅ Numba acceleration complete!")
