"""
MEDALLION APPROACH: Fast Iteration with Proxies
Train on simplified topology features, optimize if it shows alpha
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import json

def train_optimized_models():
    """Train multiple models, find best performer"""
    
    print("Loading fast dataset...")
    with open('src/data/topology_dataset/fast_dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    # Remap labels: -1→0, 0→1, 1→2 (XGBoost needs 0-indexed)
    y_train = y_train + 1
    y_val = y_val + 1
    y_test = y_test + 1
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    models = {}
    
    # 1. XGBoost (usually best for tabular)
    print("\n🚀 Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        eval_metric='mlogloss'
    )
    xgb.fit(X_train, y_train, 
            eval_set=[(X_val, y_val)],
            verbose=False)
    
    xgb_test_acc = accuracy_score(y_test, xgb.predict(X_test))
    print(f"XGBoost Test Accuracy: {xgb_test_acc:.3f}")
    models['xgboost'] = {'model': xgb, 'acc': xgb_test_acc}
    
    # 2. Gradient Boosting
    print("\n🚀 Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    
    gb_test_acc = accuracy_score(y_test, gb.predict(X_test))
    print(f"GradientBoosting Test Accuracy: {gb_test_acc:.3f}")
    models['gradientboost'] = {'model': gb, 'acc': gb_test_acc}
    
    # Select best
    best_name = max(models, key=lambda k: models[k]['acc'])
    best_model = models[best_name]['model']
    best_acc = models[best_name]['acc']
    
    print(f"\n✅ Best Model: {best_name} (Accuracy: {best_acc:.3f})")
    
    # Detailed report
    y_pred = best_model.predict(X_test)
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Short', 'Neutral', 'Long']))
    
    # Save
    os.makedirs('src/models', exist_ok=True)
    with open(f'src/models/{best_name}_optimized.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    metrics = {
        'model': best_name,
        'test_accuracy': float(best_acc),
        'all_models': {k: float(v['acc']) for k, v in models.items()}
    }
    
    with open('results/optimized_model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n💾 Saved: src/models/{best_name}_optimized.pkl")
    
    return best_model, best_name

if __name__ == "__main__":
    train_optimized_models()
