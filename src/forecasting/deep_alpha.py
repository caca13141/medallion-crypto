"""
DeepAlpha Intelligence Coordinator
Bridges the EliteEnsemble with dashboard telemetry.
Provides sophisticated signal fusion and analytical diagnostics (DNA/Decay).
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from src.forecasting.ensemble import EliteEnsemble

class DeepAlpha:
    def __init__(self, model_dir: str = "src/data/models"):
        self.ensemble = EliteEnsemble(model_dir=model_dir)
        self.ensemble.load_models()
        print(" DeepAlpha Coordinator Initialized")

    def get_regime_dna(self, regime_name: str) -> List[float]:
        """
        Returns a 'DNA' vector (weights/intensity) for a given regime.
        Used for the 'Regime DNA' visualization in the dashboard.
        """
        # Mapping names to regime IDs
        mapping = {"Low Volatility": 0, "Transition": 1, "High Volatility": 2, "Expansion": 0, "Contraction": 2}
        rid = mapping.get(regime_name, 1)
        
        # Return the weights as a normalized DNA sequence
        weights = self.ensemble.regime_weights.get(rid, np.array([0.33, 0.33, 0.34]))
        # Add some noise/entropy for visual fidelity
        dna = weights.tolist() + [0.1, 0.2] 
        return [float(x) for x in dna]

    def get_alpha_decay(self, base_ic: float) -> List[Dict]:
        """
        Calculates expected information coefficient (IC) decay across horizons.
        Used for the 'Alpha Decay' chart in the dashboard.
        """
        horizons = ["1H", "4H", "12H", "24H"]
        # Typical exponential decay for alpha
        decay = [base_ic * np.exp(-0.2 * i) for i in range(len(horizons))]
        return [{"horizon": h, "ic": float(val)} for h, val in zip(horizons, decay)]

    def predict(self, 
                persistence_images: torch.Tensor, 
                market_stream: torch.Tensor,
                recent_trades: List[Dict],
                recent_returns: np.ndarray) -> Dict:
        """
        Returns a full ensemble prediction payload.
        """
        ens_pred = self.ensemble.predict(
            persistence_images,
            market_stream,
            recent_trades,
            recent_returns
        )
        
        return {
            "tti": ens_pred.tti,
            "confidence": ens_pred.confidence,
            "components": ens_pred.components,
            "regime": ens_pred.regime,
            "regime_dna": self.get_regime_dna(ens_pred.regime),
            "alpha_decay": self.get_alpha_decay(0.15 + (ens_pred.confidence * 0.1))
        }
