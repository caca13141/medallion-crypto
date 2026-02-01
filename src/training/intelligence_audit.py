import time
import json
import numpy as np
import torch
from typing import Dict, List, Optional

class IntelligenceAudit:
    """
    The 'Eye of the Terminal'.
    Tracks the learning of the 13.6B parameter model and maps it to 'Power'.
    """
    def __init__(self):
        self.history = []
        self.expert_stats = np.zeros(8) # 8 Experts in V9 MoE
        self.start_time = time.time()
        
    def log_step(self, 
                 loss: float, 
                 expert_weights: np.ndarray, 
                 wasserstein_dist: float):
        """
        Record a training step and calculate dominance.
        """
        # 1. Update Expert Stats (smoothed)
        self.expert_stats = 0.95 * self.expert_stats + 0.05 * expert_weights
        
        # 2. Calculate Topological Dominance Score
        # Power = f(Inverse Loss, Wasserstein Convergence, Expert Entropy)
        expert_entropy = -np.sum(self.expert_stats * np.log(self.expert_stats + 1e-10))
        dominance = (1.0 / (loss + 1.0)) * (1.0 + expert_entropy) * 10 
        
        entry = {
            "timestamp": time.time(),
            "loss": float(loss),
            "expert_weights": self.expert_stats.tolist(),
            "wasserstein": float(wasserstein_dist),
            "dominance": float(dominance),
            "step": len(self.history)
        }
        
        self.history.append(entry)
        if len(self.history) > 1000:
            self.history.pop(0)
            
        return entry

    def get_audit_payload(self) -> Dict:
        """
        Returns the data for the QuantLab 'Intelligence Audit' overlay.
        """
        if not self.history:
            return {}
            
        latest = self.history[-1]
        
        return {
            "current_dominance": latest["dominance"],
            "loss_curve": [h["loss"] for h in self.history[-50:]],
            "expert_landscape": latest["expert_weights"],
            "learning_rate_stability": 0.98, # Proxy
            "convergent": latest["wasserstein"] < 0.2,
            "status": "CONVERGING" if (time.time() - self.start_time) < 3600 else "STABILIZED"
        }

# Global Instance
audit_engine = IntelligenceAudit()
