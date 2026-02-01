"""
Historical Data Collection for Model Training
Fetches BTC/USDC data from Binance (free, no auth required for historical data)
"""
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from pathlib import Path

def fetch_historical_ohlcv(exchange, symbol, timeframe='1m', start_date='2023-01-01', end_date='2025-01-01'):
    """
    Fetch OHLCV data for training
    """
    print(f" Fetching {symbol} {timeframe} data from {start_date} to {end_date}...")
    
    # Parse dates
    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    end = exchange.parse8601(f"{end_date}T00:00:00Z")
    
    all_ohlcv = []
    current = since
    
    while current < end:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current, limit=1000)
            if not ohlcv:
                break
                
            all_ohlcv.extend(ohlcv)
            current = ohlcv[-1][0] + 1  # Next timestamp
            
            print(f"  Fetched up to {datetime.fromtimestamp(current/1000)} ({len(all_ohlcv)} candles)")
            time.sleep(exchange.rateLimit / 1000)  # Respect rate limits
            
        except Exception as e:
            print(f" Error: {e}")
            time.sleep(5)
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    return df

def fetch_historical_trades(exchange, symbol, start_date, end_date, max_trades=1000000):
    """
    Fetch raw trades (for Hawkes training)
    Note: This is limited for free tier, use OHLCV for most models
    """
    print(f" Fetching recent trades for {symbol}...")
    
    try:
        trades = exchange.fetch_trades(symbol, limit=1000)
        df = pd.DataFrame(trades)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df[['timestamp', 'price', 'amount', 'side']]
    except Exception as e:
        print(f" Trade fetch not available: {e}")
        return None

def compute_returns(df, window=100):
    """
    Compute returns for Regime Detector training
    """
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Rolling volatility
    df['volatility'] = df['returns'].rolling(window).std()
    
    # Remove NaNs
    df = df.dropna()
    
    return df

def main():
    """
    Collect 2 years of BTC/USDC data for training
    """
    # Initialize exchange (no auth needed for public data)
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    symbol = 'BTC/USDC'
    
    # Fetch OHLCV (1-minute candles)
    print("\n" + "="*60)
    print("PHASE 1: OHLCV Data Collection")
    print("="*60)
    
    df_ohlcv = fetch_historical_ohlcv(
        exchange, 
        symbol, 
        timeframe='1m',
        start_date='2023-01-01',
        end_date='2025-01-01'
    )
    
    # Save raw OHLCV
    output_path = Path('data/raw/btc_ohlcv_2023_2025.parquet')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_ohlcv.to_parquet(output_path, compression='gzip')
    
    print(f"\n Saved {len(df_ohlcv)} candles to {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Compute Returns for Regime Detector
    print("\n" + "="*60)
    print("PHASE 2: Feature Engineering (Returns)")
    print("="*60)
    
    df_returns = compute_returns(df_ohlcv, window=100)
    
    # Save features
    features_path = Path('data/features/price_returns.parquet')
    features_path.parent.mkdir(parents=True, exist_ok=True)
    df_returns.to_parquet(features_path, compression='gzip')
    
    print(f" Saved returns to {features_path}")
    
    # Create train/val/test splits
    print("\n" + "="*60)
    print("PHASE 3: Dataset Splits")
    print("="*60)
    
    # 2023-2024 = Train (80%)
    # Q4 2024 = Validation (10%)
    # 2025 = Test (10%)
    
    train_end = '2024-10-01'
    val_end = '2024-12-31'
    
    train_df = df_returns[df_returns['timestamp'] < train_end]
    val_df = df_returns[(df_returns['timestamp'] >= train_end) & (df_returns['timestamp'] < val_end)]
    test_df = df_returns[df_returns['timestamp'] >= val_end]
    
    splits_dir = Path('data/splits')
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Save split indices
    train_df.to_parquet(splits_dir / 'train_2023_2024.parquet')
    val_df.to_parquet(splits_dir / 'val_2024_q4.parquet')
    test_df.to_parquet(splits_dir / 'test_2025.parquet')
    
    print(f" Train: {len(train_df)} samples ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f" Val:   {len(val_df)} samples ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")
    print(f" Test:  {len(test_df)} samples ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")
    
    print("\n" + "="*60)
    print(" DATA COLLECTION COMPLETE")
    print("="*60)
    print(f"Total samples: {len(df_returns)}")
    print(f"Date range: {df_returns['timestamp'].min()} to {df_returns['timestamp'].max()}")
    print(f"\nNext: Run train_regime_detector.py")

if __name__ == "__main__":
    main()
