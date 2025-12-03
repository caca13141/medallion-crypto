import sys
import os
import logging
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config import Config
from src.data.feed import HyperliquidFeed
from src.rl.trading_env import TradingEnv
from src.alpha.factors.momentum import calculate_ema, calculate_rsi
from src.alpha.factors.volatility import calculate_atr
from src.alpha.factors.structure import calculate_zigzag
from src.alpha.factors.persistence import calculate_hurst
from src.alpha.factors.narrative import NarrativeMapper

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PPO_TRAIN")

def prepare_data(coin="BTC", limit=5000):
    logger.info(f"FETCHING {limit} CANDLES FOR {coin}...")
    feed = HyperliquidFeed(Config.API_URL)
    df = feed.get_candles(coin, Config.TIMEFRAME, limit=limit)
    
    if df.empty:
        logger.error("NO DATA FETCHED")
        return None
        
    logger.info("CALCULATING FEATURES...")
    
    # 1. Momentum
    df['ema_fast'] = calculate_ema(df['c'], Config.EMA_FAST)
    df['ema_slow'] = calculate_ema(df['c'], Config.EMA_SLOW)
    df['rsi'] = calculate_rsi(df['c'], Config.RSI_PERIOD)
    
    # 2. Volatility
    df['atr'] = calculate_atr(df, Config.ATR_PERIOD)
    
    # 3. Structure
    df['zigzag'] = calculate_zigzag(df, Config.ZIGZAG_THRESHOLD)
    # Zigzag Low/High flags
    # Need to look back to find recent pivots. 
    # For training state, we need "Has Recent Low/High" at each step.
    # This is expensive to compute rolling. 
    # Simplified: If zigzag value is -1 within last 10 bars.
    df['zigzag_low'] = df['zigzag'].rolling(10).apply(lambda x: 1.0 if (x == -1).any() else 0.0)
    df['zigzag_high'] = df['zigzag'].rolling(10).apply(lambda x: 1.0 if (x == 1).any() else 0.0)
    
    # 4. Persistence (Hurst)
    # Rolling Hurst is expensive. Using window=100.
    logger.info("CALCULATING HURST (This may take a moment)...")
    df['hurst'] = df['c'].rolling(100).apply(lambda x: calculate_hurst(x))
    
    # 5. Narrative
    # NarrativeMapper needs history. We can map the whole df or rolling.
    # NarrativeMapper.map_narrative takes a df and returns ID for the LAST point.
    # We need ID for EVERY point.
    # For speed, let's just use a placeholder or simplified clustering on rolling window.
    # Or just skip narrative for training MVP if it's too slow.
    # Let's use a dummy narrative for now to speed up training test.
    df['narrative'] = 0.0 
    
    # Drop NaNs
    df.dropna(inplace=True)
    logger.info(f"DATA PREPARED: {len(df)} ROWS")
    
    return df

def train():
    df = prepare_data(limit=1000) # Small limit for quick test
    if df is None: return
    
    logger.info("INITIALIZING ENVIRONMENT...")
    env = TradingEnv(df)
    
    logger.info("INITIALIZING PPO AGENT...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=Config.PPO_LEARNING_RATE,
        gamma=Config.PPO_GAMMA,
        gae_lambda=Config.PPO_GAE_LAMBDA,
        clip_range=Config.PPO_CLIP_RANGE,
        verbose=1
    )
    
    logger.info(f"STARTING TRAINING ({Config.PPO_TRAIN_TIMESTEPS} TIMESTEPS)...")
    model.learn(total_timesteps=Config.PPO_TRAIN_TIMESTEPS)
    
    logger.info("SAVING MODEL...")
    model.save("src/data/ppo_model")
    logger.info("TRAINING COMPLETE.")

if __name__ == "__main__":
    train()
