"""
Ensemble Trading System
Switches between conservative and aggressive PPO agents based on market regime.
"""
import sys
sys.path.append('/Users/raphaelmaksoud/crypto toppo')

from src.rl.continuous_ppo import ContinuousTopoEnv
from stable_baselines3 import PPO
import numpy as np

class MarketRegimeDetector:
    """Detects market regime to select appropriate trading strategy."""
    
    def __init__(self, lookback=20):
        self.lookback = lookback
        self.volatility_history = []
        self.tti_history = []
        
    def detect_regime(self, price_history, tti_history):
        """
        Determine market regime based on volatility and topology.
        
        Returns:
            'aggressive': High volatility, strong trends, exploit opportunities
            'conservative': Low volatility, choppy, preserve capital
        """
        # Calculate recent volatility
        if len(price_history) < self.lookback:
            return 'conservative'  # Default to safe mode
        
        recent_prices = price_history[-self.lookback:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns)
        
        # Calculate TTI trend
        recent_tti = tti_history[-self.lookback:]
        tti_mean = np.mean(recent_tti)
        tti_std = np.std(recent_tti)
        
        # Regime rules
        HIGH_VOL_THRESHOLD = 0.02  # 2% volatility
        LOW_TTI_THRESHOLD = 6.0    # Stable topology
        
        if volatility > HIGH_VOL_THRESHOLD and tti_mean < LOW_TTI_THRESHOLD:
            # High volatility + stable topology = trending market
            return 'aggressive'
        elif volatility < HIGH_VOL_THRESHOLD * 0.5:
            # Low volatility = range-bound market  
            return 'conservative'
        elif tti_std > 2.0:
            # High TTI variance = unstable, be careful
            return 'conservative'
        else:
            # Mixed signals, default to conservative
            return 'conservative'

class EnsemblePPOAgent:
    """
    Ensemble agent that switches between conservative and aggressive policies.
    """
    
    def __init__(self, conservative_path, aggressive_path):
        print(" Loading ensemble agents...")
        
        # Load both models
        env = ContinuousTopoEnv()
        self.conservative = PPO.load(conservative_path, env=env)
        self.aggressive = PPO.load(aggressive_path, env=env)
        
        # Regime detector
        self.regime_detector = MarketRegimeDetector()
        
        # Tracking
        self.current_regime = 'conservative'
        self.regime_switches = 0
        
        print("   Conservative agent loaded")
        print("   Aggressive agent loaded")
        print("   Regime detector initialized")
    
    def predict(self, obs, price_history, tti_history):
        """
        Predict action using regime-appropriate agent.
        """
        # Detect regime
        regime = self.regime_detector.detect_regime(price_history, tti_history)
        
        # Track regime changes
        if regime != self.current_regime:
            self.regime_switches += 1
            print(f" Regime switch: {self.current_regime} → {regime}")
            self.current_regime = regime
        
        # Select agent
        if regime == 'aggressive':
            action, _states = self.aggressive.predict(obs, deterministic=True)
            agent_name = "AGGRESSIVE"
        else:
            action, _states = self.conservative.predict(obs, deterministic=True)
            agent_name = "CONSERVATIVE"
        
        return action, agent_name, regime

# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print(" Ensemble PPO Agent Demo")
    print("=" * 60)
    
    # Initialize ensemble (will fail until aggressive training completes)
    try:
        ensemble = EnsemblePPOAgent(
            conservative_path="src/data/ppo_final.zip",
            aggressive_path="src/data/ppo_aggressive.zip"
        )
        
        print("\n Ensemble ready!")
        print("\nRegime Detection Strategy:")
        print("   Conservative (Default):")
        print("     - Low volatility markets")
        print("     - High TTI variance (unstable)")
        print("     - Capital preservation focus")
        print("\n   Aggressive (Opportunistic):")
        print("     - High volatility + stable topology")
        print("     - Clear trending conditions")
        print("     - Maximize returns")
        
    except FileNotFoundError as e:
        print("\n  Aggressive model not found yet")
        print(f"   Waiting for training to complete...")
        print(f"   Expected: src/data/ppo_aggressive.zip")
        print("\n   Monitor: tail -f ppo_aggressive_training.log")
    
    print("\n" + "=" * 60)
