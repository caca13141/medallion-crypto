"""
Regime Detector (HMM)
Identifies market regimes (Low Vol, High Vol, Transition) to adapt forecasting models.
"""

import numpy as np
from hmmlearn import hmm
import joblib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class MarketRegime:
    id: int
    name: str
    volatility_level: float
    description: str

class RegimeDetector:
    """
    Detects market regimes using Gaussian Hidden Markov Model.
    
    Regimes:
    0: Low Volatility (Calm, Mean Reverting) -> Trust Transformer
    1: Transition (Trending/Uncertain) -> Mixed Weights
    2: High Volatility (Crash/Pump, Cascades) -> Trust Hawkes
    """
    
    def __init__(self, n_components: int = 3, model_dir: str = "src/data/models"):
        self.n_components = n_components
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=n_components, 
            covariance_type="full", 
            n_iter=100,
            random_state=42
        )
        
        self.is_fitted = False
        self.regime_map = {} # Map component ID to Regime Name based on variance
        
    def fit(self, returns: np.ndarray):
        """
        Fit HMM to historical returns
        Args:
            returns: (N_samples, 1) array of log returns
        """
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
            
        self.model.fit(returns)
        self.is_fitted = True
        
        # Identify regimes by volatility (variance)
        variances = [np.diag(self.model.covars_[i])[0] for i in range(self.n_components)]
        sorted_indices = np.argsort(variances)
        
        self.regime_map = {
            sorted_indices[0]: MarketRegime(0, "Low Volatility", variances[sorted_indices[0]], "Calm, Mean Reverting"),
            sorted_indices[1]: MarketRegime(1, "Transition", variances[sorted_indices[1]], "Trending/Uncertain"),
            sorted_indices[2]: MarketRegime(2, "High Volatility", variances[sorted_indices[2]], "Crash/Pump, Cascades")
        }
        
        print(f" HMM Fitted. Regimes sorted by vol: {variances}")
        
    def predict(self, recent_returns: np.ndarray) -> MarketRegime:
        """
        Predict current regime
        Args:
            recent_returns: (Window, 1) array of recent returns
        """
        if not self.is_fitted:
            # Default to Transition if not fitted
            return MarketRegime(1, "Transition", 0.0, "Default (Unfitted)")
            
        if recent_returns.ndim == 1:
            recent_returns = recent_returns.reshape(-1, 1)
        
        # Scale input if scaler is available
        if hasattr(self, 'scaler') and self.scaler is not None:
             # Prepare features exactly like training: [returns, volatility, volume]
             # For prediction, we might only have returns passed in. 
             # If the model expects 3 features, we can't predict with just returns.
             # Ideally train_regime_detector passed only 3 cols.
             # HACK: If dimension mismatch, pad? 
             # Better: assume caller passes pre-processed feature vector or handle it.
             # For NOW: If model expects 3 dims and we got 1, just use what we have if it fits, or warn.
             
             # Check dimension match
             if recent_returns.shape[1] == self.scaler.n_features_in_:
                 recent_returns = self.scaler.transform(recent_returns)
            
        # Predict state sequence, take the last one
        hidden_states = self.model.predict(recent_returns)
        current_state = hidden_states[-1]
        
        return self.regime_map.get(current_state, MarketRegime(1, "Unknown", 0.0, "Error"))
        
    def save(self):
        joblib.dump(self.model, self.model_dir / "hmm_model.pkl")
        joblib.dump(self.regime_map, self.model_dir / "regime_map.pkl")
        
    def load(self):
        model_path = self.model_dir / "hmm_model.pkl"
        
        print(f" Checking for model at: {model_path.absolute()}")
        
        if model_path.exists():
            import pickle
            with open(model_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # handle both dict (new format) and direct model (legacy)
            if isinstance(checkpoint, dict):
                self.model = checkpoint['model']
                self.scaler = checkpoint.get('scaler') # Load scaler if available
                # If regime map is in checkpoint, use it, otherwise use default/saved
                if 'regime_map' in checkpoint:
                     self.regime_map = checkpoint['regime_map']
                
                print(f" Regime Detector Loaded (Dict format, {checkpoint.get('n_states', 0)} states)")
            else:
                self.model = checkpoint
                print(" Regime Detector Loaded (Legacy format)")
                
            # Fit dummy data to initialize if needed or set flag
            self.is_fitted = True
            
            # Load standalone regime map if exists and not in checkpoint
            map_path = self.model_dir / "regime_map.pkl"
            if map_path.exists() and not self.regime_map:
                try:
                    self.regime_map = joblib.load(map_path)
                except:
                    pass
                    
        else:
            print(f" Regime Detector model not found at {model_path}")

if __name__ == "__main__":
    print(" Testing RegimeDetector...")
    
    detector = RegimeDetector()
    
    # Mock Returns (Mixture of Gaussians)
    np.random.seed(42)
    returns_low = np.random.normal(0, 0.001, 100)
    returns_high = np.random.normal(0, 0.02, 50)
    returns_mixed = np.concatenate([returns_low, returns_high, returns_low])
    
    detector.fit(returns_mixed)
    
    current = detector.predict(returns_mixed[-10:])
    print(f"Current Regime: {current.name} (Vol: {current.volatility_level:.6f})")
