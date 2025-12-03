import pandas as pd
import numpy as np
from src.config import Config

class FusionFactor:
    """
    Calculates Fusion Score based on On-chain/Microstructure data.
    """
    
    def calculate_fusion_score(self, df, funding_df, current_oi, coin_index):
        """
        df: Price candles
        funding_df: Historical funding rates
        current_oi: Current Open Interest (raw value)
        coin_index: Index of coin in universe (to normalize OI if needed, or just use raw for now)
        """
        score = 0
        
        # 1. Funding Rate Analysis
        # High positive funding = Longs paying Shorts = Crowded Longs -> Bearish
        # High negative funding = Shorts paying Longs = Crowded Shorts -> Bullish (Squeeze potential)
        
        current_funding = 0.0
        if not funding_df.empty:
            # Funding df usually has 'fundingRate'
            current_funding = float(funding_df.iloc[-1]['fundingRate'])
            
        # Normalize funding (e.g. > 0.01% is high)
        if current_funding > Config.FUNDING_THRESHOLD:
            score -= 1 # Crowded Longs
        elif current_funding < -Config.FUNDING_THRESHOLD:
            score += 1 # Crowded Shorts (Squeeze potential)
            
        # 2. Open Interest Analysis
        # We need historical OI to calculate Z-Score, but we only have current OI from snapshot.
        # For MVP, we'll just check if OI is "high" based on a static heuristic or if we can build history.
        # Since we don't have OI history in this MVP, we'll use a simplified logic:
        # If Funding is Negative AND Price is Rising -> Short Squeeze -> Bullish
        
        # Price Trend (Last 5 candles)
        price_change = (df.iloc[-1]['c'] - df.iloc[-5]['c']) / df.iloc[-5]['c']
        
        if current_funding < 0 and price_change > 0.01:
            score += 2 # Strong Short Squeeze Signal
            
        if current_funding > 0 and price_change < -0.01:
            score -= 2 # Long Liquidation Cascade Signal
            
        return score, current_funding
