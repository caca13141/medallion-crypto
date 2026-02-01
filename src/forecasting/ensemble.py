"""
Elite Ensemble Forecaster
Combines orthogonal signals from Topology, NeuralCDE, and Hawkes processes.
Uses a meta-learner (Ridge Regression) to stack predictions.
"""

import numpy as np
import torch
import torch.nn as nn
import torchcde
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import joblib
from pathlib import Path

# Import individual models
from src.forecasting.topology_forecaster import TopoTransformerGPT, create_model as create_topo_model
from src.signals.signature_cde import NeuralCDEPredictor
from src.forecasting.regime_detector import RegimeDetector

@dataclass
class EnsemblePrediction:
    tti: float
    price_direction: float
    volatility: float
    confidence: float
    components: Dict[str, float]
    regime: str

class HawkesPredictor:
    """Placeholder for Hawkes Process Predictor"""
    def predict(self, events: List[Dict]) -> float:
        return 0.5

class EliteEnsemble:
    """
    Meta-Learner for stacking orthogonal forecasts.
    Regime-aware weighting.
    """
    
    def __init__(self, model_dir: str = "src/data/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Base Models
        self.topo_model = create_topo_model()
        self.cde_model = NeuralCDEPredictor(input_channels=7, hidden_channels=32, output_channels=1)
        self.hawkes_model = HawkesPredictor()
        self.regime_detector = RegimeDetector(model_dir=str(self.model_dir))
        
        # Regime-Specific Weights (Topo, CDE, Hawkes)
        # 0: Low Vol -> Trust Topo (Patterns)
        # 1: Transition -> Balanced
        # 2: High Vol -> Trust Hawkes (Cascades)
        self.regime_weights = {
            0: np.array([0.6, 0.2, 0.2]),
            1: np.array([0.33, 0.33, 0.33]),
            2: np.array([0.2, 0.3, 0.5])
        }
        self.bias = 0.0
        
        self.is_fitted = False
        
    def load_models(self):
        """Load pre-trained base models"""
        # Load Topo Model
        topo_path = self.model_dir / "topo_transformer.pt"
        if topo_path.exists():
            self.topo_model.load_state_dict(torch.load(topo_path))
            self.topo_model.eval()
            print(" Loaded Topology Transformer")
            
        # Load CDE Model
        cde_path = self.model_dir / "neural_cde.pt"
        if cde_path.exists():
            self.cde_model.load_state_dict(torch.load(cde_path))
            self.cde_model.eval()
            print(" Loaded NeuralCDE")
            
        # Load Regime Detector
        self.regime_detector.load()
            
    def predict(self, 
                persistence_images: torch.Tensor, 
                market_stream: torch.Tensor,
                recent_trades: List[Dict],
                recent_returns: Optional[np.ndarray] = None) -> EnsemblePrediction:
        """
        Generate ensemble prediction
        """
        # 1. Topology Prediction
        with torch.no_grad():
            topo_out, _, _ = self.topo_model(persistence_images)
            topo_tti = topo_out[0, 1].item()
            
        # 2. NeuralCDE Prediction
        # 2. NeuralCDE Prediction
        try:
            with torch.no_grad():
                B, S, C = market_stream.shape
                # Ensure float32
                market_stream = market_stream.float()
                times = torch.linspace(0, S-1, S).to(market_stream.device).float()
                
                # Coefficients
                coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(market_stream, t=times)
                cde_out = self.cde_model(coeffs)
                cde_val = cde_out[0, 0].item()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f" NeuralCDE Failed: {e}")
            cde_val = 0.0
            
        # 3. Hawkes Prediction
        hawkes_val = self.hawkes_model.predict(recent_trades)
        
        # 4. Regime Detection
        regime_name = "Unknown"
        weights = self.regime_weights[1] # Default to Transition
        
        if recent_returns is not None and self.regime_detector.is_fitted:
            regime = self.regime_detector.predict(recent_returns)
            regime_name = regime.name
            weights = self.regime_weights.get(regime.id, weights)
        
        # 5. Ensemble Aggregation
        components = np.array([topo_tti, cde_val, hawkes_val])
        ensemble_val = np.dot(weights, components) + self.bias
        
        # Confidence estimation
        divergence = np.var(components)
        confidence = 1.0 / (1.0 + divergence)
        
        return EnsemblePrediction(
            tti=float(ensemble_val),
            price_direction=0.0,
            volatility=0.0,
            confidence=float(confidence),
            components={
                "topo": float(topo_tti),
                "cde": float(cde_val),
                "hawkes": float(hawkes_val)
            },
            regime=regime_name
        )
    
    def fit_meta_learner(self, X_preds, y_actuals):
        # Placeholder for fitting logic
        pass

if __name__ == "__main__":
    print(" Testing EliteEnsemble...")
    ensemble = EliteEnsemble()
    
    # Mock Inputs
    B, S = 1, 72
    mock_images = torch.randn(B, S, 1, 32, 32)
    mock_stream = torch.randn(B, S, 7)
    mock_trades = [{"price": 100, "time": 1}]
    mock_returns = np.random.randn(100)
    
    # Predict
    pred = ensemble.predict(mock_images, mock_stream, mock_trades, mock_returns)
    print(f"Prediction: {pred}")
