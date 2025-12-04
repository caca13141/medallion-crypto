#!/usr/bin/env python3
"""
MAXIMUM TRAINING - REAL DATA
Trains on 207k samples until convergence
"""

import torch
import numpy as np
import pickle
from torch.utils.data import TensorDataset, DataLoader
from src.training.max_learning import MaxLearningPipeline
from xgboost import XGBClassifier
import json

print("🔥 MAXIMUM TRAINING ON REAL DATA")
print("="*60)

# Load REAL topology features
print("Loading 207k samples...")
with open('src/data/topology_dataset/numba_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']

print(f"✅ Train: {len(X_train):,}")
print(f"✅ Val: {len(X_val):,}")
print(f"✅ Test: {len(X_test):,}")

# Strategy 1: Train XGBoost FIRST (proven to work)
print("\n" + "="*60)
print("STRATEGY 1: XGBoost (Proven)")
print("="*60)

xgb = XGBClassifier(
    n_estimators=500,  # More trees
    max_depth=8,       # Deeper
    learning_rate=0.03,  # Slower (better)
    random_state=42,
    tree_method='hist',  # Faster
    eval_metric='mlogloss'
)

# Remap labels for XGBoost
y_train_xgb = y_train + 1
y_val_xgb = y_val + 1

print("Training XGBoost (500 trees, max convergence)...")
xgb.fit(
    X_train, y_train_xgb,
    eval_set=[(X_val, y_val_xgb)],
    verbose=50
)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report

y_pred = xgb.predict(X_test)
test_acc = accuracy_score(y_test + 1, y_pred)

print(f"\n✅ XGBoost Test Accuracy: {test_acc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test + 1, y_pred, target_names=['Short', 'Neutral', 'Long']))

# Save
xgb.save_model('models/xgboost_207k.json')

# Backtest
print("\n" + "="*60)
print("BACKTESTING XGBoost")
print("="*60)

from src.backtest.backtester import Backtester
import pandas as pd

# Load prices
df = pd.read_parquet('src/data/historical/btc_usdt_15m.parquet')
lookback = 50
forecast = 24

test_start_idx = int((len(df) - lookback - forecast) * 0.8) + lookback
test_end_idx = len(df) - forecast
prices_test = df.iloc[test_start_idx:test_end_idx]['close'].values[:len(X_test)]

backtester = Backtester(xgb, initial_capital=10000)
metrics, equity_curve, trades = backtester.backtest(X_test, y_test + 1, prices_test)

print("\n📊 BACKTEST RESULTS (207k samples)")
print("="*60)
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")

# Save results
with open('results/xgboost_207k_backtest.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Plot equity
backtester.plot_equity_curve(equity_curve, 'results/equity_207k.png')

print("\n" + "="*60)
print("🏁 TRAINING COMPLETE")
print("="*60)
print(f"""
✅ Trained on 207,527 samples (6 years)
✅ Test Accuracy: {test_acc:.1%}
✅ Backtest Return: {metrics['total_return_pct']:.2f}%
✅ Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
✅ Win Rate: {metrics['win_rate']:.1%}

Models saved:
- models/xgboost_207k.json
- results/xgboost_207k_backtest.json
- results/equity_207k.png

Next: If profitable → Train 36-Layer Transformer overnight
""")
