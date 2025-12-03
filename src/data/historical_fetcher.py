"""
Historical Data Fetcher for TopoOmega Training
Fetches 2 years of Hyperliquid candles for backtesting and training
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from src.data.feed import HyperliquidFeed
from src.config import Config
import pickle
import os

class HistoricalFetcher:
    """
    Fetch and store historical data for training.
    """
    
    def __init__(self, coins=['BTC', 'ETH', 'SOL'], start_date='2023-01-01'):
        self.feed = HyperliquidFeed(Config.API_URL)
        self.coins = coins
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime('2025-12-03')
        
    def fetch_historical(self, coin, interval='15m', limit=1000):
        """
        Fetch historical candles for a coin.
        Hyperliquid API returns max ~1000 candles per call, so we need to paginate.
        """
        print(f"Fetching historical data for {coin}...")
        
        all_candles = []
        current_end = int(self.end_date.timestamp() * 1000)
        
        # Work backwards from end date
        while True:
            try:
                # Calculate start time for this batch
                if interval == '15m':
                    ms_per_candle = 15 * 60 * 1000
                elif interval == '1h':
                    ms_per_candle = 60 * 60 * 1000
                else:
                    ms_per_candle = 15 * 60 * 1000
                    
                current_start = current_end - (limit * ms_per_candle)
                
                # Stop if we've reached start date
                if current_start < int(self.start_date.timestamp() * 1000):
                    current_start = int(self.start_date.timestamp() * 1000)
                
                # Fetch batch
                candles = self.feed.info.candles_snapshot(
                    coin, interval, current_start, current_end
                )
                
                if not candles:
                    break
                    
                all_candles.extend(candles)
                print(f"  Fetched {len(candles)} candles (total: {len(all_candles)})")
                
                # Check if we've reached start
                if current_start <= int(self.start_date.timestamp() * 1000):
                    break
                    
                # Move window back
                current_end = current_start - 1
                
                # Rate limit
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  Error: {e}")
                time.sleep(2)
                continue
        
        # Convert to DataFrame
        if all_candles:
            df = pd.DataFrame(all_candles)
            
            # Parse columns
            df['t'] = pd.to_datetime(df['t'], unit='ms')
            for col in ['c', 'o', 'h', 'l', 'v']:
                df[col] = df[col].astype(float)
            
            # Sort by time
            df = df.sort_values('t').reset_index(drop=True)
            
            print(f"✅ Total {len(df)} candles from {df['t'].min()} to {df['t'].max()}")
            return df
        else:
            print(f"❌ No data fetched")
            return pd.DataFrame()
    
    def fetch_all(self, interval='15m'):
        """Fetch for all coins and save"""
        os.makedirs('src/data/historical', exist_ok=True)
        
        for coin in self.coins:
            df = self.fetch_historical(coin, interval)
            
            if not df.empty:
                # Save to parquet (efficient storage)
                filename = f'src/data/historical/{coin.lower()}_{interval}.parquet'
                df.to_parquet(filename)
                print(f"💾 Saved {coin} to {filename}")
            else:
                print(f"⚠️  No data for {coin}")
            
            time.sleep(1)  # Rate limit between coins
        
        print("\n✅ Historical data fetch complete!")

if __name__ == "__main__":
    fetcher = HistoricalFetcher(coins=['BTC'], start_date='2024-01-01')
    fetcher.fetch_all(interval='15m')
