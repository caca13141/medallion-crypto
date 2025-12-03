"""
Simple Baseline Trainer
Trains on topology features to predict 6h direction
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import json

def train_baseline():
    """Train simple baseline on topology features"""
    
    # Load dataset
    print("Loading dataset...")
    with open('src/data/topology_dataset/fast_dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    
    print(f"Train: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    val_acc = accuracy_score(y_val, rf.predict(X_val))
    test_acc = accuracy_score(y_test, rf.predict(X_test))
    
    print(f"Random Forest Results:")
    print(f"  Train Accuracy: {train_acc:.3f}")
    print(f"  Val Accuracy: {val_acc:.3f}")
    print(f"  Test Accuracy: {test_acc:.3f}")
    
    # Detailed metrics
    y_pred_test = rf.predict(X_test)
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['Short', 'Neutral', 'Long']))
    
    # Feature importance
    importances = rf.feature_importances_
    feature_names = ['Volatility', 'Momentum', 'AC_1', 'AC_5', 'AC_10', 
                     'Vol_Change', 'HL_Range', 'Max_Up', 'Max_Down', 'Avg_Abs_Return']
    
    print("\nFeature Importances:")
    for name, imp in zip(feature_names, importances):
        print(f"  {name}: {imp:.3f}")
    
    # Save model
    os.makedirs('src/models', exist_ok=True)
    with open('src/models/baseline_rf.pkl', 'wb') as f:
        pickle.dump(rf, f)
    
    # Save metrics
    metrics = {
        'model': 'RandomForest',
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        'feature_importances': {name: float(imp) for name, imp in zip(feature_names, importances)}
    }
    
    with open('results/baseline_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Model saved to src/models/baseline_rf.pkl")
    print(f"✅ Metrics saved to results/baseline_metrics.json")
    
    return rf, metrics

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    train_baseline()
