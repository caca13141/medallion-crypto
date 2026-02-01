"""
Train NeuralCDE on Market Stream Data
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
sys.path.append('src')

from signals.signature_cde import NeuralCDEPredictor

class MarketStreamDataset(Dataset):
    """Dataset for continuous-time market data"""
    def __init__(self, streams, targets):
        self.streams = torch.FloatTensor(streams)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.streams)
    
    def __getitem__(self, idx):
        return self.streams[idx], self.targets[idx]

def prepare_market_streams(df, window=72):
    """
    Prepare 7-dimensional market streams
    [mid_price, spread, vol_imb, returns, volatility, volume_change, regime]
    """
    streams = []
    targets = []
    
    # Compute features
    df = df.copy()
    df['mid'] = df['close']  # Simplified
    df['spread'] = 0.001  # Mock
    df['vol_imb'] = 0.0  # Mock
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(100).std()
    if 'volume' in df.columns:
        df['vol_change'] = df['volume'].pct_change()
    else:
        df['vol_change'] = 0.0
    df['regime'] = 0  # Mock (would use HMM here)
    
    # Normalize
    features = ['mid', 'spread', 'vol_imb', 'returns', 'volatility', 'vol_change', 'regime']
    for feat in features:
        df[feat] = df[feat].fillna(0)
        if feat != 'regime':
            df[feat] = (df[feat] - df[feat].mean()) / (df[feat].std() + 1e-8)
        df[feat] = df[feat].clip(-5, 5)
    
    # Create windows
    for i in range(window, len(df) - 10):
        stream = df[features].iloc[i-window:i].values
        target = df['close'].iloc[i+10] / df['close'].iloc[i] - 1  # 10-step return
        target = np.clip(target, -0.1, 0.1)
        
        streams.append(stream)
        targets.append(target)
    
    return np.array(streams), np.array(targets)

def train_neural_cde():
    print("="*60)
    print("NEURALCDE TRAINING")
    print("="*60)
    
    # Load data
    print("\n Loading training data...")
    train_df = pd.read_parquet('data/splits/train_2023_2024.parquet')
    val_df = pd.read_parquet('data/splits/val_2024_q4.parquet')
    
    # Prepare streams
    print(" Preparing market streams...")
    train_streams, train_targets = prepare_market_streams(train_df, window=72)
    val_streams, val_targets = prepare_market_streams(val_df, window=72)
    
    print(f"   Train: {len(train_streams)} samples")
    print(f"   Val:   {len(val_streams)} samples")
    
    # Create datasets
    train_dataset = MarketStreamDataset(train_streams, train_targets)
    val_dataset = MarketStreamDataset(val_streams, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Initialize model
    print("\n Initializing NeuralCDE...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    model = NeuralCDEPredictor(
        input_channels=7,
        hidden_channels=64,
        output_dim=1
    ).to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # Train
    print("\n Training...")
    epochs = 30
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for streams, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            streams = streams.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(streams)
            loss = criterion(predictions.squeeze(), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for streams, targets in val_loader:
                streams = streams.to(device)
                targets = targets.to(device)
                
                predictions = model(streams)
                loss = criterion(predictions.squeeze(), targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step()
        
        print(f"   Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_path = Path('src/data/models/neural_cde.pt')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, output_path)
            print(f"    Saved best model (val_loss={val_loss:.6f})")
    
    print("\n" + "="*60)
    print(" NEURALCDE TRAINING COMPLETE")
    print("="*60)
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Model: src/data/models/neural_cde.pt")

if __name__ == "__main__":
    train_neural_cde()
