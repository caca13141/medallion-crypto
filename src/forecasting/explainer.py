"""
Feature Attribution (SHAP)
Explains which topological features drive the model's predictions.
"""

import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path
from src.forecasting.topology_forecaster import TopoTransformerGPT, create_model

class ForecastExplainer:
    """
    Explains model predictions using SHAP (SHapley Additive exPlanations).
    Focuses on identifying which parts of the persistence diagram (H0/H1) 
    contribute most to the TTI forecast.
    """
    
    def __init__(self, model_path: str = "src/data/models/topo_transformer.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model().to(self.device)
        
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f" Loaded model for explanation from {model_path}")
        else:
            print(" Model not found, using initialized weights for testing")
            
        # Background dataset for SHAP (initialized lazily)
        self.background_data = None
        self.explainer = None
        
    def prepare_background(self, images: torch.Tensor, n_samples: int = 100):
        """
        Prepare background dataset for SHAP (DeepExplainer or GradientExplainer).
        Args:
            images: (B, Seq, 1, 32, 32) tensor of historical persistence images
        """
        # Randomly sample background
        indices = np.random.choice(len(images), min(len(images), n_samples), replace=False)
        self.background_data = images[indices].to(self.device)
        
        # Initialize DeepExplainer
        # Note: DeepExplainer can be memory intensive. 
        # We wrap the model to output only the scalar TTI prediction.
        self.explainer = shap.DeepExplainer(self.ModelWrapper(self.model), self.background_data)
        print(" SHAP Explainer Initialized")
        
    class ModelWrapper(torch.nn.Module):
        """Wraps model to output only TTI scalar for SHAP"""
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            # x: (B, Seq, 1, 32, 32)
            # Output: (B, 1) -> TTI
            scalars, _, _ = self.model(x)
            return scalars[:, 1:2] # TTI is index 1
            
    def explain(self, recent_images: torch.Tensor) -> Dict:
        """
        Compute SHAP values for recent images.
        Args:
            recent_images: (B, Seq, 1, 32, 32)
        Returns:
            Dict with shap_values and feature_importance
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call prepare_background() first.")
            
        recent_images = recent_images.to(self.device)
        
        # Compute SHAP values
        # shap_values will be list of tensors (one per output)
        # Since we output only TTI, it's a list of length 1
        shap_values = self.explainer.shap_values(recent_images)
        
        # shap_values[0] shape: (B, Seq, 1, 32, 32)
        # We want to aggregate importance
        
        # 1. Temporal Importance (Which time steps matter?)
        # Sum absolute SHAP values across spatial dims
        temporal_importance = np.abs(shap_values).sum(axis=(2, 3, 4)) # (B, Seq)
        temporal_importance = temporal_importance.mean(axis=0) # (Seq,)
        
        # 2. Spatial Importance (Which parts of the PD matter?)
        # Sum absolute SHAP values across time
        spatial_importance = np.abs(shap_values).sum(axis=1) # (B, 1, 32, 32)
        spatial_importance = spatial_importance.mean(axis=0).squeeze() # (32, 32)
        
        return {
            "shap_values": shap_values,
            "temporal_importance": temporal_importance.tolist(),
            "spatial_importance": spatial_importance.tolist()
        }
        
    def plot_importance(self, importance_dict: Dict, save_path: str = "shap_plot.png"):
        """Visualize SHAP importance"""
        spatial = np.array(importance_dict["spatial_importance"])
        temporal = np.array(importance_dict["temporal_importance"])
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Temporal
        ax[0].plot(temporal)
        ax[0].set_title("Temporal Importance (Last 72h)")
        ax[0].set_xlabel("Hours Ago")
        ax[0].set_ylabel("Mean |SHAP|")
        
        # Spatial (Persistence Image)
        im = ax[1].imshow(spatial, origin='lower', cmap='viridis')
        ax[1].set_title("Topological Feature Importance")
        ax[1].set_xlabel("Birth")
        ax[1].set_ylabel("Persistence")
        plt.colorbar(im, ax=ax[1])
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f" SHAP plot saved to {save_path}")
        plt.close()

if __name__ == "__main__":
    print(" Testing ForecastExplainer...")
    
    explainer = ForecastExplainer()
    
    # Mock Data
    B, S = 10, 72
    mock_images = torch.randn(B, S, 1, 32, 32)
    
    # Prepare
    explainer.prepare_background(mock_images[:5])
    
    # Explain
    explanation = explainer.explain(mock_images[5:6])
    
    print(f"Temporal Importance Shape: {len(explanation['temporal_importance'])}")
    print(f"Spatial Importance Shape: {np.array(explanation['spatial_importance']).shape}")
    
    explainer.plot_importance(explanation, "src/data/shap_test.png")
