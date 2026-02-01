"""
MONTE CARLO FORECASTING ENGINE
Generates probability distributions for price predictions using bootstrap resampling
and uncertainty quantification through multiple simulation paths.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import deque
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ProbabilityForecast:
    """Probabilistic forecast with confidence intervals."""
    timestamp: float
    p5: float   # 5th percentile
    p25: float  # 25th percentile (Q1)
    p50: float  # 50th percentile (median)
    p75: float  # 75th percentile (Q3)
    p95: float  # 95th percentile
    mean: float
    std: float

class MonteCarloForecaster:
    """
    Monte Carlo simulation engine for probabilistic price forecasting.
    """
    def __init__(self, 
                 model=None,
                 n_simulations=1000,
                 noise_scale=0.015,  # 1.5% noise
                 device='cpu'):
        self.model = model
        self.n_simulations = n_simulations
        self.noise_scale = noise_scale
        self.device = device
        
        # Historical volatility buffer for adaptive noise
        self.volatility_buffer = deque(maxlen=100)
        
    def estimate_volatility(self, prices: List[float]) -> float:
        """Estimate realized volatility from recent price history."""
        if len(prices) < 2:
            return self.noise_scale
        
        returns = np.diff(np.log(prices))
        volatility = np.std(returns)
        
        # Store for adaptive scaling
        self.volatility_buffer.append(volatility)
        
        return volatility
    
    def bootstrap_features(self, 
                          topology_features: np.ndarray,
                          n_samples: int) -> np.ndarray:
        """
        Bootstrap resample topology features with replacement.
        Adds controlled noise to simulate uncertainty.
        """
        n_features = topology_features.shape[0]
        
        # Bootstrap indices
        indices = np.random.choice(n_features, size=(n_samples, n_features), replace=True)
        
        # Resample
        bootstrapped = topology_features[indices]
        
        # Add Gaussian noise scaled by feature variance
        noise = np.random.randn(*bootstrapped.shape) * self.noise_scale
        bootstrapped = bootstrapped + noise
        
        return bootstrapped
    
    def generate_price_paths(self,
                            base_price: float,
                            trend_strength: float,
                            volatility: float,
                            horizon: int = 48,
                            roughness: float = 0.5,
                            adversarial_bias: float = 0.0) -> np.ndarray:
        """
        'Frontier V8' Engine: Adversarial Heston-Merton with Signature Roughness.
        """
        dt = 1.0 / 24.0
        n_sims = self.n_simulations
        
        # 1. Signature Roughness Scaling
        # roughness (Hurst proxy) < 0.5 implies mean-reversion/roughness
        # roughness > 0.5 implies persistence
        vol_scale = 1.0 + (0.5 - roughness) * 2.0 # Increase vol if rough
        kappa = 2.0 * vol_scale; theta = volatility; xi = 0.15 * vol_scale; rho = -0.7
        
        # 2. Adversarial Predatory Drift
        # adversarial_bias represents the "counter-party" push
        
        # Merton Jump Constants
        lambda_j = 0.1 * vol_scale; mu_j = -0.01; sigma_j = 0.05
        
        # Pre-allocate
        paths = np.zeros((n_sims, horizon))
        paths[:, 0] = base_price
        
        Z1 = np.random.randn(n_sims, horizon-1)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.randn(n_sims, horizon-1)
        Jumps = np.random.poisson(lambda_j * dt, (n_sims, horizon-1))
        JumpSizes = Jumps * (mu_j + sigma_j * np.random.randn(n_sims, horizon-1))
        
        curr_p = np.full(n_sims, base_price)
        curr_v = np.full(n_sims, volatility)
        
        for t in range(1, horizon):
            dv = kappa * (theta - curr_v) * dt + xi * np.sqrt(np.maximum(curr_v, 1e-5)) * np.sqrt(dt) * Z2[:, t-1]
            curr_v = np.maximum(curr_v + dv, 1e-5)
            
            # Apply Adversarial Bias to drift
            # If we are long, adversary pushes down. If short, pushes up.
            predatory_drift = -adversarial_bias * 0.001 
            
            drift = (trend_strength + predatory_drift - 0.5 * curr_v) * dt
            diffusion = np.sqrt(curr_v) * np.sqrt(dt) * Z1[:, t-1]
            curr_p = curr_p * np.exp(drift + diffusion + JumpSizes[:, t-1])
            paths[:, t] = curr_p
            
        return paths

    def calculate_greeks(self, paths: np.ndarray, last_price: float, adversarial_paths: np.ndarray = None) -> Dict[str, float]:
        """
        Calculates Greeks and Adversarial VaR (AVaR).
        """
        final_prices = paths[:, -1]
        delta = np.mean(final_prices > last_price)
        
        from scipy.stats import kurtosis
        gamma = kurtosis(final_prices, fisher=True)
        
        # Standard VaR
        var_95 = np.percentile(final_prices, 5)
        cvar_95 = final_prices[final_prices <= var_95].mean() if len(final_prices[final_prices <= var_95]) > 0 else var_95
        
        # Adversarial VaR (AVaR)
        avar_95 = var_95
        if adversarial_paths is not None:
            adv_final = adversarial_paths[:, -1]
            avar_95 = np.percentile(adv_final, 5)
        
        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "var_95": float((var_95 - last_price) / last_price * 100),
            "cvar_95": float((cvar_95 - last_price) / last_price * 100),
            "avar_95": float((avar_95 - last_price) / last_price * 100)
        }
    
    def compute_confidence_intervals(self,
                                    price_paths: np.ndarray,
                                    timestamps: List[float]) -> List[ProbabilityForecast]:
        """
        Compute percentiles across simulation paths.
        
        Args:
            price_paths: Array of shape (n_simulations, horizon)
            timestamps: List of Unix timestamps for each horizon step
            
        Returns:
            List of ProbabilityForecast objects
        """
        forecasts = []
        
        for t_idx, timestamp in enumerate(timestamps):
            prices_at_t = price_paths[:, t_idx]
            
            forecast = ProbabilityForecast(
                timestamp=timestamp,
                p5=float(np.percentile(prices_at_t, 5)),
                p25=float(np.percentile(prices_at_t, 25)),
                p50=float(np.percentile(prices_at_t, 50)),
                p75=float(np.percentile(prices_at_t, 75)),
                p95=float(np.percentile(prices_at_t, 95)),
                mean=float(np.mean(prices_at_t)),
                std=float(np.std(prices_at_t))
            )
            
            forecasts.append(forecast)
        
        return forecasts
    
    def forecast(self,
                current_price: float,
                price_history: List[float],
                topology_signature,
                horizon: int = 48) -> List[ProbabilityForecast]:
        """
        Generate probabilistic forecast for the next `horizon` hours.
        
        Args:
            current_price: Current market price
            price_history: Recent price history (for volatility estimation)
            topology_signature: TopologySignature object
            horizon: Forecast horizon in hours
            
        Returns:
            List of ProbabilityForecast objects, one per hour
        """
        # Estimate market volatility
        volatility = self.estimate_volatility(price_history)
        
        # Extract trend signal from topology
        # Higher loop score = stronger trend
        # Higher TTI = more reversal risk
        loop_score = topology_signature.loop_score if hasattr(topology_signature, 'loop_score') else 0
        tti = topology_signature.tti if hasattr(topology_signature, 'tti') else 0
        
        # Trend strength: positive loop score drives upward, negative reversal
        trend_strength = (loop_score * 0.001) - (tti * 0.0005)
        
        # Generate price paths
        price_paths = self.generate_price_paths(
            base_price=current_price,
            trend_strength=trend_strength,
            volatility=volatility,
            horizon=horizon
        )
        
        # Generate timestamps
        from datetime import datetime, timedelta
        base_time = datetime.now()
        timestamps = [(base_time + timedelta(hours=h)).timestamp() for h in range(horizon)]
        
        # Compute confidence intervals
        forecasts = self.compute_confidence_intervals(price_paths, timestamps)
        
        return forecasts
    
    def assess_confidence(self, forecast: ProbabilityForecast) -> str:
        """
        Assess forecast confidence based on interval width.
        
        Args:
            forecast: ProbabilityForecast object
            
        Returns:
            Confidence label: 'HIGH', 'MEDIUM', 'LOW'
        """
        # Relative spread
        spread = (forecast.p95 - forecast.p5) / forecast.p50
        
        if spread < 0.05:  # < 5% spread
            return 'HIGH'
        elif spread < 0.10:  # 5-10% spread
            return 'MEDIUM'
        else:
            return 'LOW'

# Institutional Alias
MonteCarloEngine = MonteCarloForecaster

# Example Usage
if __name__ == "__main__":
    # Simulate
    forecaster = MonteCarloEngine(n_simulations=100)
    
    # Mock data
    current_price = 92000.0
    price_history = [92000 + np.random.randn() * 500 for _ in range(100)]
    
    # Mock topology signature
    class MockTopo:
        loop_score = 2.5
        tti = 1.5
    
    forecasts = forecaster.forecast(current_price, price_history, MockTopo(), horizon=48)
    
    # Print sample
    print("Monte Carlo Forecast (48h):")
    print(f"Current: ${current_price:,.0f}")
    print(f"\nIn 24h:")
    f24 = forecasts[24]
    print(f"  Median: ${f24.p50:,.0f}")
    print(f"  95% CI: [${f24.p5:,.0f}, ${f24.p95:,.0f}]")
    print(f"  Confidence: {forecaster.assess_confidence(f24)}")
    
    print(f"\nIn 48h:")
    f48 = forecasts[47]
    print(f"  Median: ${f48.p50:,.0f}")
    print(f"  95% CI: [${f48.p5:,.0f}, ${f48.p95:,.0f}]")
    print(f"  Confidence: {forecaster.assess_confidence(f48)}")
