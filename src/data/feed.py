import time
import pandas as pd
from hyperliquid.info import Info
from src.core.logger import setup_logger

logger = setup_logger("DATA_FEED")

class HyperliquidFeed:
    def __init__(self, api_url):
        self.info = Info(api_url, skip_ws=True)
        
    def get_universe(self, limit=30):
        try:
            meta = self.info.meta()
            universe = meta['universe']
            return [u['name'] for u in universe[:limit]]
        except Exception as e:
            logger.error(f"Universe Fetch Error: {e}")
            return []
            
    def get_candles(self, coin, interval, limit=100):
        try:
            # Hyperliquid expects ms timestamps
            end_time = int(time.time() * 1000)
            start_time = end_time - (limit * 15 * 60 * 1000 * 2) # Buffer
            
            candles = self.info.candles_snapshot(coin, interval, start_time, end_time)
            if not candles: return pd.DataFrame()
            
            df = pd.DataFrame(candles)
            cols = ['c', 'o', 'h', 'l', 'v']
            for c in cols: df[c] = df[c].astype(float)
            return df
        except Exception as e:
            logger.error(f"Candle Fetch Error {coin}: {e}")
            return pd.DataFrame()
            
    def get_user_state(self, address):
        try:
            return self.info.user_state(address)
        except Exception as e:
            logger.error(f"User State Error: {e}")
            return None

    def get_funding_history(self, coin, limit=100):
        try:
            end_time = int(time.time() * 1000)
            start_time = end_time - (limit * 3600 * 1000) # 1 hour per funding rate usually
            funding = self.info.funding_history(coin, start_time, end_time)
            return pd.DataFrame(funding)
        except Exception as e:
            logger.error(f"Funding Fetch Error {coin}: {e}")
            return pd.DataFrame()

    def get_asset_ctxs(self):
        try:
            meta, asset_ctxs = self.info.meta_and_asset_ctxs()
            return asset_ctxs
        except Exception as e:
            logger.error(f"Asset Ctx Fetch Error: {e}")
            return []
