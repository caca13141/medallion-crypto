import sys
import os
import logging
import pandas as pd
import numpy as np
import itertools

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config import Config
from src.data.feed import HyperliquidFeed
from src.alpha.signal_engine import SignalEngine
from src.risk.sizing import RiskManager

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RISK_OPTIMIZER")

class RiskOptimizer:
    def __init__(self):
        self.feed = HyperliquidFeed(Config.API_URL)
        self.engine = SignalEngine()
        
    def fetch_data(self, coin="BTC", limit=1000):
        logger.info(f"FETCHING {limit} CANDLES FOR {coin}...")
        df = self.feed.get_candles(coin, Config.TIMEFRAME, limit=limit)
        return df

    def simulate(self, df, risk_per_trade, max_leverage):
        """
        Simulate trading with specific risk parameters.
        Returns: Sharpe Ratio, Max Drawdown, Final Equity
        """
        equity = 10000.0 # Start with $10k simulation
        initial_equity = equity
        peak_equity = equity
        max_drawdown = 0.0
        
        position = 0 # 0=Neutral, 1=Long, -1=Short
        entry_price = 0.0
        size_token = 0.0
        
        returns = []
        
        # Pre-calculate signals to speed up simulation
        # In a real optimization, we might want to re-calculate signals if they depended on equity,
        # but our signals are price-based. Sizing depends on equity.
        
        # We need to run the loop to update equity dynamically
        
        # Pre-calculate indicators
        # We'll use the engine's analyze method but we need to do it row by row or pre-calc
        # For speed, let's pre-calculate indicators on the whole DF first
        # But SignalEngine.analyze does this inside.
        # Let's iterate row by row.
        
        # To speed up, we'll assume signals are already generated or generate them once
        # But SignalEngine is stateful (Transformer model).
        # For this simulation, let's just use the rule-based part or run analyze every step.
        # Running analyze 1000 times is fine.
        
        for i in range(Config.LOOKBACK, len(df)):
            # Slice window
            window = df.iloc[i-Config.LOOKBACK:i+1].copy()
            current_price = window.iloc[-1]['c']
            
            # Generate Signal
            # We pass None for feed/coin to skip Fusion/On-chain for speed in simulation
            signal, atr, _, _, _ = self.engine.analyze(window)
            
            # PnL Calculation for existing position
            step_pnl = 0.0
            if position != 0:
                price_change = current_price - entry_price
                if position == 1: step_pnl = size_token * price_change
                else: step_pnl = size_token * -price_change
                
                # Update equity (unrealized + realized if we closed)
                # For simulation, we mark to market every step
                # Actually, let's just track realized PnL on close/flip
                pass
            
            # Execution Logic
            if signal != 0 and signal != position:
                # Close existing
                if position != 0:
                    price_change = current_price - entry_price
                    pnl = 0
                    if position == 1: pnl = size_token * price_change
                    else: pnl = size_token * -price_change
                    
                    equity += pnl
                    # Fee
                    equity -= size_token * current_price * 0.0005
                    
                    returns.append(pnl / initial_equity) # Simple return tracking
                
                # Open new
                # Calculate Size
                # Risk Manager uses Config.RISK_PER_TRADE, we need to override it
                # We'll implement a simple sizing logic here matching RiskManager but with params
                
                risk_amount = equity * risk_per_trade
                # Stop Loss distance = 2 * ATR (Hardcoded in RiskManager usually, let's assume 2)
                sl_dist = 2 * atr
                if sl_dist == 0: sl_dist = current_price * 0.01
                
                size_usd = risk_amount / (sl_dist / current_price)
                
                # Leverage Cap
                max_size = equity * max_leverage
                size_usd = min(size_usd, max_size)
                
                size_token = size_usd / current_price
                entry_price = current_price
                position = signal
                
            # Track Drawdown
            peak_equity = max(peak_equity, equity)
            dd = (peak_equity - equity) / peak_equity
            max_drawdown = max(max_drawdown, dd)
            
            if equity <= 0:
                return -1.0, 1.0, 0.0 # Bust
                
        # Close final position
        if position != 0:
            current_price = df.iloc[-1]['c']
            price_change = current_price - entry_price
            pnl = 0
            if position == 1: pnl = size_token * price_change
            else: pnl = size_token * -price_change
            equity += pnl
            
        total_return = (equity - initial_equity) / initial_equity
        
        # Sharpe Ratio (Annualized)
        # Assuming 15m candles -> 96 per day -> 35040 per year
        if len(returns) > 1:
            avg_ret = np.mean(returns)
            std_ret = np.std(returns)
            if std_ret == 0: sharpe = 0
            else: sharpe = (avg_ret / std_ret) * np.sqrt(len(returns)) # Rough approximation
        else:
            sharpe = 0
            
        return sharpe, max_drawdown, equity

    def optimize(self):
        df = self.fetch_data("BTC", limit=2000)
        if df.empty: return
        
        risks = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
        leverages = [1, 2, 3, 4, 5]
        
        results = []
        
        logger.info("STARTING GRID SEARCH...")
        for r, l in itertools.product(risks, leverages):
            sharpe, dd, final_eq = self.simulate(df, r, l)
            logger.info(f"RISK: {r*100}% | LEV: {l}x -> SHARPE: {sharpe:.2f} | DD: {dd*100:.2f}% | EQ: ${final_eq:.0f}")
            results.append({
                'risk': r,
                'leverage': l,
                'sharpe': sharpe,
                'dd': dd,
                'equity': final_eq
            })
            
        # Find Best Sharpe
        best = max(results, key=lambda x: x['sharpe'])
        logger.info("-" * 40)
        logger.info(f"OPTIMAL PARAMETERS FOUND:")
        logger.info(f"RISK PER TRADE: {best['risk']*100}%")
        logger.info(f"MAX LEVERAGE: {best['leverage']}x")
        logger.info(f"METRICS: Sharpe={best['sharpe']:.2f}, DD={best['dd']*100:.2f}%, Final Equity=${best['equity']:.0f}")
        logger.info("-" * 40)

if __name__ == "__main__":
    opt = RiskOptimizer()
    opt.optimize()
