"""
Signal Decay Analyzer
Measures how fast predictive power (IC) degrades over time.
Calculates the "Half-Life" of alpha signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from src.forecasting.validator import ForecastValidator

@dataclass
class DecayProfile:
    half_life_hours: float
    initial_ic: float
    decay_curve: Dict[int, float]  # Horizon -> IC
    is_tradable: bool  # True if half-life > min_threshold

class DecayAnalyzer:
    """
    Analyzes the temporal structure of alpha signals.
    
    Key Metric: Alpha Half-Life
    - Time it takes for the Information Coefficient (IC) to drop by 50%.
    - Short half-life (< 1h) -> HFT execution required
    - Long half-life (> 4h) -> Position trading possible
    """
    
    def __init__(self, validator: ForecastValidator):
        self.validator = validator
        
    def compute_decay_profile(self, 
                             model, 
                             data: np.ndarray, 
                             targets_by_horizon: Dict[int, np.ndarray]) -> DecayProfile:
        """
        Compute the full decay profile for a model
        """
        # 1. Get IC for each horizon
        decay_ics = self.validator.analyze_signal_decay(model, data, targets_by_horizon)
        
        if not decay_ics:
            return DecayProfile(0.0, 0.0, {}, False)
            
        # 2. Fit exponential decay curve: IC(t) = IC(0) * e^(-lambda * t)
        # Linearize: ln(IC(t)) = ln(IC(0)) - lambda * t
        horizons = sorted(decay_ics.keys())
        ics = [max(1e-4, decay_ics[h]) for h in horizons] # Avoid log(0) or negative
        
        if len(horizons) < 2:
            # Fallback for single point
            half_life = horizons[0] if ics[0] > 0 else 0.0
        else:
            # Linear regression on log IC
            x = np.array(horizons)
            y = np.log(ics)
            
            # Simple least squares
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            
            # lambda = -m
            decay_lambda = -m
            
            if decay_lambda <= 0:
                half_life = float('inf') # Signal doesn't decay (or improves)
            else:
                half_life = np.log(2) / decay_lambda
                
        initial_ic = ics[0]
        is_tradable = half_life > 0.5 # Minimum 30 min half-life for this system
        
        return DecayProfile(
            half_life_hours=float(half_life),
            initial_ic=float(initial_ic),
            decay_curve=decay_ics,
            is_tradable=is_tradable
        )

if __name__ == "__main__":
    print(" Testing DecayAnalyzer...")
    
    # Mock Validator
    validator = ForecastValidator(forecast_horizons=[1, 4, 12, 24])
    analyzer = DecayAnalyzer(validator)
    
    # Mock Model
    class MockModel:
        def predict(self, X):
            return np.random.randn(len(X))
            
    model = MockModel()
    data = np.random.randn(100, 10)
    
    # Mock Targets with decaying correlation
    targets = {}
    base_signal = model.predict(data)
    
    for h in [1, 4, 12, 24]:
        # Correlation drops as h increases
        noise = np.random.randn(100) * (h * 0.5)
        targets[h] = base_signal + noise
        
    profile = analyzer.compute_decay_profile(model, data, targets)
    
    print(f"Initial IC: {profile.initial_ic:.3f}")
    print(f"Half-Life: {profile.half_life_hours:.1f} hours")
    print(f"Decay Curve: {profile.decay_curve}")
    print(f"Tradable: {profile.is_tradable}")
