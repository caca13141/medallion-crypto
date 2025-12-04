"""
Quick backtest on XGBoost model
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import pickle
from src.backtest.backtester import Backtester

# Load XGBoost model
with open('src/models/xgboost_optimized.pkl', 'rb') as f:
    model = pickle.load(f)

# Load test data
with open('src/data/topology_dataset/fast_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

X_test = data['X_test']
y_test = data['y_test'] + 1  # Remap

# Load prices
df = pd.read_parquet('src/data/historical/btc_15m.parquet')
lookback = 50
forecast = 24
test_start_idx = int((len(df) - lookback - forecast) * 0.8) + lookback
test_end_idx = len(df) - forecast
prices_test = df.iloc[test_start_idx:test_end_idx]['c'].values[:len(X_test)]

# Backtest
backtester = Backtester(model, initial_capital=10000)
metrics, equity_curve, trades = backtester.backtest(X_test, y_test, prices_test)

# Print
print("\n" + "="*60)
print("XGBOOST BACKTEST RESULTS")
print("="*60)
for key, value in metrics.items():
    print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
print("="*60)

# Save
import json
with open('results/xgboost_backtest.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Done!")
