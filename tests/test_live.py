import sys
import os
import logging
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.data.feed import HyperliquidFeed
from src.alpha.signal_engine import SignalEngine

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TEST_LIVE")

def test_live():
    logger.info("CONNECTING TO HYPERLIQUID MAINNET...")
    feed = HyperliquidFeed(Config.API_URL)
    engine = SignalEngine()
    
    coin = "BTC"
    logger.info(f"FETCHING LIVE CANDLES FOR {coin}...")
    df = feed.get_candles(coin, Config.TIMEFRAME)
    
    if df.empty:
        logger.error("FAILED TO FETCH DATA")
        return
        
    logger.info(f"FETCHED {len(df)} CANDLES. LAST PRICE: {df.iloc[-1]['c']}")
    
    logger.info("RUNNING SIGNAL ENGINE ON LIVE DATA...")
    # Pass feed and coin for Fusion
    signal, atr, narrative, hurst, fusion = engine.analyze(df, feed, coin)
    
    sig_str = "NEUTRAL"
    if signal == 1: sig_str = "LONG"
    elif signal == -1: sig_str = "SHORT"
    
    logger.info(f"LIVE RESULT {coin}:")
    logger.info(f"SIGNAL: {sig_str} ({signal})")
    logger.info(f"ATR: {atr:.2f}")
    logger.info(f"NARRATIVE ID: {narrative}")
    logger.info(f"HURST: {hurst:.2f} ({'TRENDING' if hurst > Config.HURST_THRESHOLD else 'MEAN REVERTING'})")
    logger.info(f"FUSION SCORE: {fusion}")
    
    # Also check ETH
    coin = "ETH"
    logger.info(f"\nFETCHING LIVE CANDLES FOR {coin}...")
    df = feed.get_candles(coin, Config.TIMEFRAME)
    signal, atr, narrative, hurst, fusion = engine.analyze(df, feed, coin)
    
    sig_str = "NEUTRAL"
    if signal == 1: sig_str = "LONG"
    elif signal == -1: sig_str = "SHORT"
    
    logger.info(f"LIVE RESULT {coin}:")
    logger.info(f"SIGNAL: {sig_str} ({signal})")
    logger.info(f"ATR: {atr:.2f}")
    logger.info(f"NARRATIVE ID: {narrative}")
    logger.info(f"HURST: {hurst:.2f}")
    logger.info(f"FUSION SCORE: {fusion}")

if __name__ == "__main__":
    test_live()
