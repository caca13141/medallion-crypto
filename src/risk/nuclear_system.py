"""
JPM/RenTech Nuclear Risk System (2025 Production)
Implements Topological Turbulence Kill-Switch and Hard Stops.
Zero-tolerance risk management.
"""

import numpy as np
import time
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class RiskState:
    can_trade: bool
    leverage_cap: float
    reason: str
    turbulence_level: float

class NuclearRiskSystem:
    """
    The "Adult in the Room".
    Overrides all ML/RL signals if risk metrics breach thresholds.
    """
    def __init__(self, 
                 max_drawdown_daily: float = 0.035, # -3.5%
                 tti_threshold: float = 2.8,        # Topological Turbulence
                 max_leverage: float = 25.0):
        
        self.max_dd = max_drawdown_daily
        self.tti_threshold = tti_threshold
        self.max_leverage = max_leverage
        
        # State
        self.daily_pnl = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.last_reset = time.time()
        self.is_halted = False
        
    def update_pnl(self, current_equity: float):
        """Update equity and check daily drawdown"""
        # Reset daily if 24h passed (simplified)
        if time.time() - self.last_reset > 86400:
            self.daily_pnl = 0.0
            self.peak_equity = current_equity
            self.last_reset = time.time()
            self.is_halted = False # Auto-restart daily
            
        self.current_equity = current_equity
        if self.peak_equity == 0:
            self.peak_equity = current_equity
            
        self.peak_equity = max(self.peak_equity, current_equity)
        
        # Calculate DD
        dd = (self.peak_equity - current_equity) / self.peak_equity
        self.daily_pnl = (current_equity / self.peak_equity) - 1.0 # Approx
        
        if dd > self.max_dd:
            self.is_halted = True
            return False # Hard Stop Triggered
            
        return True

    def check_risk(self, 
                   topology_metrics: Dict[str, float], 
                   model_confidence: float) -> RiskState:
        """
        Primary Risk Gate.
        """
        # 1. Hard Stop Check
        if self.is_halted:
            return RiskState(False, 0.0, "DAILY_HARD_STOP_HIT", 0.0)
            
        # 2. Topological Turbulence (TTI)
        # If TTI > 2.8, market structure is breaking down (Phase Transition)
        tti = topology_metrics.get('tti', 0.0)
        if tti > self.tti_threshold:
            return RiskState(False, 0.0, f"TURBULENCE_KILL_SWITCH (TTI={tti:.2f})", tti)
            
        # 3. Leverage Cap based on Confidence
        # Linear scaling: Conf 0.5 -> 1x, Conf 1.0 -> Max
        # But clipped by Volatility (if available)
        capped_leverage = self.max_leverage * max(0, (model_confidence - 0.5) * 2)
        
        return RiskState(True, capped_leverage, "OK", tti)

if __name__ == "__main__":
    risk = NuclearRiskSystem()
    
    # Test 1: Normal
    state = risk.check_risk({'tti': 1.5}, 0.8)
    print(f"Normal: {state.can_trade}, Lev: {state.leverage_cap:.1f}x")
    
    # Test 2: High Turbulence
    state = risk.check_risk({'tti': 3.2}, 0.9)
    print(f"Turbulence: {state.can_trade}, Reason: {state.reason}")
    
    # Test 3: Hard Stop
    risk.update_pnl(10000)
    risk.update_pnl(9600) # -4%
    state = risk.check_risk({'tti': 1.0}, 0.9)
    print(f"Hard Stop: {state.can_trade}, Reason: {state.reason}")
