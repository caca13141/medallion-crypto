"""
Nuclear Risk Controls: Topological Kill-Switch + Confidence Caps
"""
import numpy as np

class NuclearRiskControls:
    """
    Advanced risk management using topology and prediction confidence.
    
    Features:
    1. Topological turbulence kill-switch
    2. Prediction confidence leverage cap
    3. Daily drawdown hard stop (-3.5%)
    4. Auto-restart after recovery
    """
    
    def __init__(
        self,
        tti_threshold=3.0,
        confidence_min=0.6,
        daily_dd_limit=0.035,
        min_leverage=1,
        max_leverage=25
    ):
        self.tti_threshold = tti_threshold
        self.confidence_min = confidence_min
        self.daily_dd_limit = daily_dd_limit
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage
        
        # State tracking
        self.daily_start_equity = None
        self.is_flatted = False
        self.last_reset_time = None
        
    def should_flatten(self, tti, daily_equity, start_equity):
        """
        Determine if should flatten all positions.
        
        Returns:
            (should_flatten: bool, reason: str)
        """
        # 1. Topological turbulence trigger
        if tti > self.tti_threshold:
            self.is_flatted = True
            return True, f"TTI_BREACH ({tti:.2f} > {self.tti_threshold})"
            
        # 2. Daily drawdown trigger
        if start_equity is not None:
            dd = (start_equity - daily_equity) / start_equity
            if dd > self.daily_dd_limit:
                self.is_flatted = True
                return True, f"DAILY_DD ({dd*100:.2f}% > {self.daily_dd_limit*100}%)"
                
        return False, None
        
    def calculate_leverage_cap(self, confidence, base_leverage):
        """
        Cap leverage based on prediction confidence.
        
        Args:
            confidence: model confidence [0, 1]
            base_leverage: requested leverage
            
        Returns:
            capped_leverage: actual leverage to use
        """
        # If confidence too low, force to 1x
        if confidence < self.confidence_min:
            return self.min_leverage
            
        # Scale leverage by confidence
        # confidence=0.6 → 50% of base
        # confidence=1.0 → 100% of base
        confidence_scale = (confidence - self.confidence_min) / (1.0 - self.confidence_min)
        capped = base_leverage * confidence_scale
        
        # Clamp
        capped = max(self.min_leverage, min(self.max_leverage, capped))
        
        return capped
        
    def should_resume_trading(self, tti, current_equity, start_equity):
        """
        Determine if should resume trading after being flatted.
        
        Resume conditions:
        1. TTI back below threshold - 0.5
        2. Equity recovered to within 2% of daily start
        """
        if not self.is_flatted:
            return True
            
        # Check TTI
        tti_ok = tti < (self.tti_threshold - 0.5)
        
        # Check equity recovery
        equity_ok = True
        if start_equity is not None:
            dd = (start_equity - current_equity) / start_equity
            equity_ok = dd < 0.02  # Within 2% of start
            
        # Resume if both OK
        if tti_ok and equity_ok:
            self.is_flatted = False
            return True
            
        return False
        
    def reset_daily(self, current_equity):
        """Reset daily tracking (call at start of trading day)"""
        self.daily_start_equity = current_equity
        # Don't reset is_flatted - it persists until conditions improve
        
    def get_max_position_size(self, equity, price, atr, confidence, leverage):
        """
        Calculate maximum position size with all risk controls.
        
        Args:
            equity: current account equity
            price: current price
            atr: average true range (volatility)
            confidence: model confidence
            leverage: desired leverage
            
        Returns:
            size_usd: maximum position size in USD
        """
        # If flatted, return 0
        if self.is_flatted:
            return 0.0
            
        # Apply confidence-based leverage cap
        capped_leverage = self.calculate_leverage_cap(confidence, leverage)
        
        # Risk per trade: 0.5% of equity
        risk_per_trade = 0.005
        risk_amount = equity * risk_per_trade
        
        # Stop loss: 2 × ATR
        stop_loss_distance = 2 * atr
        if stop_loss_distance == 0:
            stop_loss_distance = price * 0.01  # Fallback: 1%
            
        # Position size based on risk
        size_usd = risk_amount / (stop_loss_distance / price)
        
        # Apply leverage cap
        max_size = equity * capped_leverage
        size_usd = min(size_usd, max_size)
        
        return size_usd
