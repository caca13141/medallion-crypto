"""
Train Regime Detector
Fetches historical BTC data and trains the HMM to detect market regimes.
"""

import sys
import os
import numpy as np
import pandas as pd
from src.data.historical_fetcher import HistoricalFetcher
from src.forecasting.regime_detector import RegimeDetector

def train():
    print(" Starting Regime Detector Training...")
    
    # 1. Fetch Data
    # Fetching 1 month of 15m data (approx 3000 candles) is enough for HMM
    fetcher = HistoricalFetcher(coins=['BTC'], start_date='2024-11-01')
    df = fetcher.fetch_historical('BTC', interval='15m', limit=3000)
    
    if df.empty:
        print(" No data fetched. Aborting.")
        return
        
    # 2. Preprocess
    # Log returns
    df['return'] = np.log(df['c'] / df['c'].shift(1))
    returns = df['return'].dropna().values.reshape(-1, 1)
    
    print(f" Training on {len(returns)} samples...")
    
    # 3. Train HMM
    detector = RegimeDetector()
    detector.fit(returns)
    
    # 4. Save
    detector.save()
    
    # 5. Verify
    print("\n Verification:")
    print("Regimes identified:")
    for i, regime in detector.regime_map.items():
        print(f"  Regime {i}: {regime.name} (Vol: {regime.volatility_level:.6f})")
        
    # Test on last 10 samples
    last_window = returns[-10:]
    current_regime = detector.predict(last_window)
    print(f"\nCurrent Regime (based on last 10 candles): {current_regime.name}")

if __name__ == "__main__":
    train()
