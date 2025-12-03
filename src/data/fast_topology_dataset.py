"""
FAST Topology Dataset Generator (Simplified)
Focuses on key metrics instead of full persistence diagrams for speed
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

class FastTopologyDataset:
    """Simplified topology dataset for quick training"""
    
    def __init__(self, lookback=50):  # Reduced from 100
        self.lookback = lookback
        
    def compute_simple_features(self, window):
        """
        Fast approximations instead of full persistence:
        - Price volatility (proxy for H0 complexity)
        - Cycle detection via autocorrelation (proxy for H1 loops)
        - Trend strength
        """
        prices = window['c'].values
        
        if len(prices) < 10:
            return np.zeros(10)
        
        # Returns
        returns = np.diff(np.log(prices))
        
        # Features
        vol = np.std(returns)
        momentum = np.mean(returns[-10:])
        
        # Autocorrelation (cycle proxy)
        if len(returns) > 20:
            ac_1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            ac_5 = np.corrcoef(returns[:-5], returns[5:])[0, 1]
            ac_10 = np.corrcoef(returns[:-10], returns[10:])[0, 1]
        else:
            ac_1, ac_5, ac_10 = 0, 0, 0
        
        # Volume features
        vol_change = np.mean(np.diff(window['v'].values[-10:]))
        
        # Range features
        hl_range = np.mean((window['h'].values - window['l'].values) / window['c'].values)
        
        return np.array([
            vol,
            momentum,
            ac_1,
            ac_5,
            ac_10,
            vol_change,
            hl_range,
            np.max(returns),  # Max up move
            np.min(returns),  # Max down move
            np.mean(np.abs(returns))  # Average absolute return
        ])
    
    def generate(self, parquet_path):
        """Generate dataset quickly"""
        print(f"Loading {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        features_list = []
        labels_list = []
        
        print("Generating features (simplified, fast mode)...")
        
        for i in tqdm(range(self.lookback, len(df) - 24)):  # 24 candles = 6h forecast
            window = df.iloc[i-self.lookback:i]
            
            features = self.compute_simple_features(window)
            
            # Label: next 6h return
            future_return = (df.iloc[i+24]['c'] / df.iloc[i]['c']) - 1
            label = 1 if future_return > 0.01 else (-1 if future_return < -0.01 else 0)
            
            features_list.append(features)
            labels_list.append(label)
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"\n✅ Generated {len(X)} samples")
        print(f"Feature shape: {X.shape}")
        print(f"Label distribution: {np.bincount(y + 1)}")  # +1 to handle -1, 0, 1
        
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
        with open('src/data/topology_dataset/fast_dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
            
        print(f"\n💾 Saved to src/data/topology_dataset/fast_dataset.pkl")
        
        return dataset

if __name__ == "__main__":
    generator = FastTopologyDataset(lookback=50)
    dataset = generator.generate('src/data/historical/btc_15m.parquet')
    print("\n✅ Fast dataset ready!")
