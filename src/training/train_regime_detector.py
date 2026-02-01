"""
Train Regime Detector (HMM) on Historical Data
This is the first model to train - it provides regime labels for other models
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def load_training_data():
    """Load prepared training data"""
    print(" Loading training data...")
    
    train_df = pd.read_parquet('data/splits/train_2023_2024.parquet')
    val_df = pd.read_parquet('data/splits/val_2024_q4.parquet')
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    
    return train_df, val_df

def prepare_features(df, window=100):
    """
    Prepare features for HMM
    - Returns
    - Volatility
    - Volume
    """
    features = []
    
    # Log returns (clip extreme values)
    returns = df['log_returns'].values.reshape(-1, 1)
    returns = np.clip(returns, -0.1, 0.1)  # Clip to ±10% to handle outliers
    features.append(returns)
    
    # Volatility (rolling std, clip to reasonable range)
    volatility = df['volatility'].values.reshape(-1, 1)
    volatility = np.clip(volatility, 0, 0.1)  # Clip volatility to max 10%
    features.append(volatility)
    
    # Volume change (if available)
    if 'volume' in df.columns:
        vol_change = df['volume'].pct_change().fillna(0).values.reshape(-1, 1)
        vol_change = np.clip(vol_change, -1, 1)  # Clip to ±100%
        features.append(vol_change)
    
    X = np.hstack(features)
    
    # Remove any remaining NaNs or Infs
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    
    print(f"  Feature shape: {X.shape}")
    print(f"  Removed {(~mask).sum()} rows with inf/nan")
    
    return X

def train_hmm_regime_detector(X_train, n_states=4, n_iter=1000):
    """
    Train HMM on returns
    
    States:
    0: Low Volatility Uptrend
    1: Low Volatility Downtrend
    2: High Volatility Uptrend
    3: High Volatility Downtrend
    """
    print(f"\n Training HMM with {n_states} states...")
    print(f"   Feature dimensions: {X_train.shape}")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train HMM
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=42,
        verbose=True
    )
    
    model.fit(X_train_scaled)
    
    print(f"\n Training complete!")
    print(f"   Converged: {model.monitor_.converged}")
    print(f"   Log Likelihood: {model.score(X_train_scaled):.2f}")
    
    return model, scaler

def interpret_states(model, scaler, X):
    """
    Interpret what each state represents
    """
    print("\n Interpreting Regime States...")
    
    X_scaled = scaler.transform(X)
    states = model.predict(X_scaled)
    
    # Analyze each state
    for state in range(model.n_components):
        mask = states == state
        state_X = X[mask]
        
        print(f"\n  State {state}:")
        print(f"    Count: {mask.sum()} ({100*mask.sum()/len(X):.1f}%)")
        print(f"    Mean Return: {state_X[:, 0].mean():.6f}")
        print(f"    Mean Volatility: {state_X[:, 1].mean():.6f}")
        if state_X.shape[1] > 2:
            print(f"    Mean Volume Change: {state_X[:, 2].mean():.6f}")
    
    return states

def validate_model(model, scaler, X_val):
    """
    Validate on held-out data
    """
    print("\n Validating on Q4 2024 data...")
    
    X_val_scaled = scaler.transform(X_val)
    val_score = model.score(X_val_scaled)
    val_states = model.predict(X_val_scaled)
    
    print(f"  Validation Log Likelihood: {val_score:.2f}")
    print(f"  Unique states: {len(np.unique(val_states))}/{model.n_components}")
    
    return val_score, val_states

def save_model(model, scaler, output_path='src/data/models/hmm_model.pkl'):
    """
    Save trained model and scaler
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'n_states': model.n_components
        }, f)
    
    print(f"\n Model saved to {output_path}")

def plot_regimes(states, df, output_path='data/features/regime_visualization.png'):
    """
    Visualize regime changes over time
    """
    print("\n Creating regime visualization...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # Plot price with regime background
    timestamps = df['timestamp'].values[:len(states)]
    prices = df['close'].values[:len(states)]
    
    ax1.plot(timestamps, prices, linewidth=0.5, alpha=0.7)
    ax1.set_ylabel('BTC Price (USDC)')
    ax1.set_title('BTC Price with Regime Detection')
    ax1.grid(True, alpha=0.3)
    
    # Color background by regime
    for state in range(len(np.unique(states))):
        mask = states == state
        ax1.fill_between(timestamps, prices.min(), prices.max(), 
                         where=mask, alpha=0.2, label=f'Regime {state}')
    
    ax1.legend(loc='upper left')
    
    # Plot regime states
    ax2.plot(timestamps, states, linewidth=0.5)
    ax2.set_ylabel('Regime State')
    ax2.set_xlabel('Date')
    ax2.set_title('Detected Market Regimes')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    print(f"  Saved to {output_path}")

def main():
    print("="*60)
    print("REGIME DETECTOR TRAINING")
    print("="*60)
    
    # Load data
    train_df, val_df = load_training_data()
    
    # Prepare features
    X_train = prepare_features(train_df)
    X_val = prepare_features(val_df)
    
    # Train HMM
    model, scaler = train_hmm_regime_detector(
        X_train,
        n_states=4,  # 4 market regimes
        n_iter=1000
    )
    
    # Interpret states
    train_states = interpret_states(model, scaler, X_train)
    
    # Validate
    val_score, val_states = validate_model(model, scaler, X_val)
    
    # Save model
    save_model(model, scaler)
    
    # Visualize (on subset for speed)
    plot_regimes(train_states[-10000:], train_df.iloc[-10000:])
    
    print("\n" + "="*60)
    print(" REGIME DETECTOR TRAINING COMPLETE")
    print("="*60)
    print(f"Model: src/data/models/hmm_model.pkl")
    print(f"States: {model.n_components}")
    print(f"Features: {X_train.shape[1]}")
    print(f"\nNext: Use this model in feed_dashboard.py (already integrated!)")

if __name__ == "__main__":
    main()
