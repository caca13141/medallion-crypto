import sys
import os
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.alpha.signal_engine import SignalEngine
from src.config import Config

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TEST_SIGNAL")

def generate_synthetic_data(length=100, trend='UP'):
    # Generate synthetic OHLCV data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=length, freq='15min')
    
    base_price = 100.0
    prices = []
    
    for i in range(length):
        noise = np.random.normal(0, 0.5)
        if trend == 'UP':
            base_price += 0.2 + noise
        elif trend == 'DOWN':
            base_price -= 0.2 + noise
        else:
            base_price += noise
            
        prices.append(base_price)
        
    df = pd.DataFrame(index=dates)
    df['c'] = prices
    df['o'] = df['c'].shift(1).fillna(prices[0])
    df['h'] = df[['c', 'o']].max(axis=1) + 0.5
    df['l'] = df[['c', 'o']].min(axis=1) - 0.5
    df['v'] = np.random.randint(100, 1000, size=length)
    
    return df

def test_engine():
    logger.info("INITIALIZING SIGNAL ENGINE...")
    engine = SignalEngine()
    
    logger.info("GENERATING SYNTHETIC UPTREND DATA...")
    df_up = generate_synthetic_data(100, trend='UP')
    
    logger.info("ANALYZING UPTREND...")
    signal, atr, narrative, hurst, fusion = engine.analyze(df_up)
    logger.info(f"RESULT: Signal={signal} (1=LONG), ATR={atr:.2f}, Narrative={narrative}, Hurst={hurst:.2f}, Fusion={fusion}")
    
    logger.info("GENERATING SYNTHETIC DOWNTREND DATA...")
    df_down = generate_synthetic_data(100, trend='DOWN')
    
    logger.info("ANALYZING DOWNTREND...")
    signal, atr, narrative, hurst, fusion = engine.analyze(df_down)
    logger.info(f"RESULT: Signal={signal} (-1=SHORT), ATR={atr:.2f}, Narrative={narrative}, Hurst={hurst:.2f}, Fusion={fusion}")

    logger.info("TESTING TRAINING STEP (WASSERSTEIN LOSS)...")
    # Train on uptrend data with target=1 (LONG)
    loss = engine.train_step(df_up, 1)
    logger.info(f"TRAINING LOSS: {loss:.4f}")


if __name__ == "__main__":
    test_engine()
