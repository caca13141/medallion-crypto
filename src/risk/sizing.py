from src.config import Config

class RiskManager:
    def calculate_size(self, equity, price, atr):
        # Stop Loss distance = 2 * ATR
        sl_dist = 2 * atr
        if sl_dist == 0: sl_dist = price * 0.01
        
        # Risk Amount = Equity * Risk%
        risk_amount = equity * Config.RISK_PER_TRADE
        
        # Position Size (USD) = Risk Amount / (SL Distance / Price)
        # Simplified: Size = Risk / %Loss
        pct_loss = sl_dist / price
        position_size_usd = risk_amount / pct_loss
        
        # Cap Leverage
        max_size = equity * Config.MAX_LEVERAGE
        return min(position_size_usd, max_size)
