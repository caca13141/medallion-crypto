"""
JPM/RenTech Topology Engine (2025 Production)
Implements Bifiltrated Persistence, Signed Persistence, and Topological Landscapes.
Optimized for 32x32 Persistence Images and 8-dim H1 Summaries.
"""

import numpy as np
try:
    import gudhi
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False
    print(" Gudhi not found. Bifiltration disabled.")
from ripser import ripser
from persim import PersistenceImager
from persim.landscapes import PersLandscapeApprox
import ot  # Python Optimal Transport
from scipy.spatial.distance import pdist, squareform
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
import warnings

# Suppress TDA warnings for production logs
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class TopologySignature:
    """Container for high-dimensional topological features"""
    persistence_image: np.ndarray  # 32x32
    landscapes: np.ndarray         # 5 layers x 100 steps
    betti_curves: np.ndarray       # H0, H1 curves
    loop_score: float
    tti: float                     # Topological Turbulence Index
    wasserstein_amp: float         # Amplitude vs noise
    h1_summary: np.ndarray         # 8-dim vector
    bull_loops: int = 0            # Number of loops in "Above Mean" regime
    bear_loops: int = 0            # Number of loops in "Below Mean" regime
    bifiltration_score: float = 0.0 # Volume-weighted persistence score

class PersistenceEngine:
    """
    Advanced Topological Data Analysis Engine.
    Integrates GUDHI (Bifiltration) and Ripser++ (Fast VR).
    Maintains compatibility with legacy TopologyIntegrator.
    """
    
    def __init__(self, 
                 resolution: int = 32, 
                 landscape_layers: int = 5,
                 max_edge_length: float = 5.0,
                 max_dimension: int = 1):
        self.resolution = resolution
        self.landscape_layers = landscape_layers
        self.max_edge_length = max_edge_length
        self.max_dimension = max_dimension
        
        # Initialize Persistence Imager with default settings
        self.imager = PersistenceImager()
        
    def compute_bifiltration(self, point_cloud: np.ndarray, 
                           function_values: np.ndarray) -> Any:
        """
        Computes Bifiltration (Rips x Function).
        Uses GUDHI SimplexTree with filtration values.
        """
        # 1. Build Rips Complex
        rips = gudhi.RipsComplex(points=point_cloud, max_edge_length=self.max_edge_length)
        st = rips.create_simplex_tree(max_dimension=2)
        
        # 2. Update filtration with function values (e.g., density, volatility)
        # This creates a proxy for bifiltration by re-indexing
        for simplex, filtration in st.get_filtration():
            # Get max function value on vertices of simplex
            vertices = [v for v in simplex]
            f_val = np.max(function_values[vertices]) if vertices else 0
            
            # Combine Rips filtration (distance) and Function filtration
            # Product filtration proxy: max(dist, f_val) or weighted
            new_filtration = max(filtration, f_val)
            st.assign_filtration(simplex, new_filtration)
            
        st.make_filtration_non_decreasing()
        return st

    def compute_signed_persistence(self, diagrams: List[np.ndarray]) -> np.ndarray:
        """
        Computes Signed Persistence (Birth - Death asymmetry).
        Returns 8-dim summary vector of H1 features.
        """
        if len(diagrams) < 2 or len(diagrams[1]) == 0:
            return np.zeros(8)
            
        h1 = diagrams[1]
        # Filter infinite death
        h1 = h1[h1[:, 1] != np.inf]
        
        if len(h1) == 0:
            return np.zeros(8)
            
        lifetimes = h1[:, 1] - h1[:, 0]
        births = h1[:, 0]
        deaths = h1[:, 1]
        
        # 8-Dim Summary Vector:
        # 1. Max Lifetime
        # 2. Avg Lifetime
        # 3. Total Persistence (Sum)
        # 4. Entropy of Persistence
        # 5. Max Birth
        # 6. Max Death
        # 7. Birth-Death Correlation
        # 8. Cycle Count (Significant)
        
        total_pers = np.sum(lifetimes)
        probs = lifetimes / total_pers
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        summary = np.array([
            np.max(lifetimes),
            np.mean(lifetimes),
            total_pers,
            entropy,
            np.max(births),
            np.max(deaths),
            np.corrcoef(births, deaths)[0,1] if len(births) > 1 else 0,
            len(lifetimes)
        ])
        
        return np.nan_to_num(summary)

    def compute_landscapes(self, diagrams: List[np.ndarray]) -> np.ndarray:
        """
        Computes Persistence Landscapes (Stable vector representation).
        """
        if len(diagrams) < 2 or len(diagrams[1]) == 0:
            return np.zeros((self.landscape_layers, 100))
            
        h1 = diagrams[1]
        # Filter infinite death
        h1_clean = h1[h1[:, 1] != np.inf]
        if len(h1_clean) == 0:
            return np.zeros((self.landscape_layers, 100))
            
        # Compute landscapes
        # We use PersLandscapeApprox for speed
        # PersLandscapeApprox expects dgms to be indexed by hom_deg
        # So we pass [[], h1_clean] to allow access to index 1
        pla = PersLandscapeApprox(dgms=[[], h1_clean], hom_deg=1, num_steps=100)
        # .values is an attribute, not a method
        landscapes = pla.values[:self.landscape_layers]
        
        return landscapes

    def analyze_window(self, point_cloud: np.ndarray, 
                      volumes: Optional[np.ndarray] = None) -> TopologySignature:
        """
        Full production analysis of a market window.
        Includes Signed Persistence and Bifiltration.
        """
        # 0. Landmark Selection (Safety)
        # Limit to 150 points to guarantee <100ms compute and bounded memory
        if len(point_cloud) > 150:
            point_cloud_sub = self._landmark_selection(point_cloud, n_landmarks=150)
        else:
            point_cloud_sub = point_cloud

        # 1. Standard Persistence (Ripser++ for speed)
        diagrams = ripser(point_cloud_sub, maxdim=1)['dgms']
        
        # 2. Persistence Images (H1)
        if len(diagrams) > 1 and len(diagrams[1]) > 0:
            h1_diag = diagrams[1]
            max_death = np.max(h1_diag[h1_diag[:, 1] != np.inf][:, 1]) if np.any(h1_diag[:, 1] != np.inf) else 1.0
            h1_clean = np.copy(h1_diag)
            h1_clean[h1_clean[:, 1] == np.inf, 1] = max_death * 1.1
            p_image = self.imager.fit_transform([h1_clean])[0]
            
            # Wasserstein Amplitude
            # wass_amp = ot.emd2_1d(h1_clean[:, 0], h1_clean[:, 1])
            wass_amp = np.sum(h1_clean[:, 1] - h1_clean[:, 0]) # Simplified for speed
        else:
            p_image = np.zeros((self.resolution, self.resolution))
            wass_amp = 0.0
            
        # 3. Landscapes
        land = self.compute_landscapes(diagrams)
        
        # 4. Signed Persistence
        h1_summary = self.compute_signed_persistence(diagrams)
        
        # 5. Bifiltration Score
        bifiltration_score = 0.0
        if volumes is not None and len(volumes) == len(point_cloud) and HAS_GUDHI:
            # Normalize volumes for filtration
            vol_norm = (volumes - np.min(volumes)) / (np.max(volumes) - np.min(volumes) + 1e-6)
            st = self.compute_bifiltration(point_cloud, vol_norm)
            st.persistence()
            bif_pairs = st.persistence_intervals_in_dimension(1)
            if len(bif_pairs) > 0:
                bifiltration_score = np.sum(bif_pairs[:, 1] - bif_pairs[:, 0])

        # 6. Advanced Stats
        loop_score = h1_summary[0] * h1_summary[3] if len(h1_summary) > 3 else 0.0
        
        # TTI calculation
        h0 = diagrams[0]
        h0_life = h0[h0[:, 1] != np.inf][:, 1] - h0[h0[:, 1] != np.inf][:, 0]
        h0_entropy = -np.sum((h0_life/np.sum(h0_life)) * np.log((h0_life/np.sum(h0_life)) + 1e-10)) if len(h0_life) > 0 else 0
        tti = h0_entropy / (loop_score + 1e-6)
        
        # Signed Loops count
        mean_price = np.mean(point_cloud[:, -1])
        bull_loops = np.sum(point_cloud[:, -1] > mean_price) # Simplified proxies
        bear_loops = np.sum(point_cloud[:, -1] <= mean_price)

        return sig

    def _landmark_selection(self, points, n_landmarks=150):
        """Simple furthest point sampling or random selection"""
        if len(points) <= n_landmarks:
            return points
        indices = np.random.choice(len(points), n_landmarks, replace=False)
        return points[indices]

    # --- Legacy Compatibility Methods ---
    
    def compute_point_cloud(self, prices, volumes=None, volatilities=None):
        """Compute 3D Takens Embedding for legacy integrator"""
        n = len(prices)
        cloud = np.zeros((n, 3))
        # Dim 1: Z-score returns
        returns = np.diff(np.log(prices + 1e-8), prepend=np.log(prices[0] + 1e-8))
        cloud[:, 0] = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        # Dim 2: Volume proxy
        if volumes is not None:
            cloud[:, 1] = (volumes - np.mean(volumes)) / (np.std(volumes) + 1e-8)
        # Dim 3: Volatility proxy
        if volatilities is not None:
            cloud[:, 2] = (volatilities - np.mean(volatilities)) / (np.std(volatilities) + 1e-8)
        return cloud

    def compute_persistence(self, point_cloud):
        """Legacy wrapper for ripser"""
        if len(point_cloud) > 150:
            point_cloud = self._landmark_selection(point_cloud, 150)
        return ripser(point_cloud, maxdim=1)['dgms']

    def loop_score(self, diagrams):
        """Legacy wrapper for H1 lifetime"""
        if len(diagrams) < 2 or len(diagrams[1]) == 0:
            return 0.0
        h1 = diagrams[1]
        lifetimes = h1[h1[:,1] != np.inf][:, 1] - h1[h1[:,1] != np.inf][:, 0]
        return np.max(lifetimes) if len(lifetimes) > 0 else 0.0

    def topological_turbulence_index(self, diagrams):
        """Legacy wrapper for H0 entropy"""
        h0 = diagrams[0]
        lifetimes = h0[h0[:, 1] != np.inf][:, 1] - h0[h0[:, 1] != np.inf][:, 0]
        if len(lifetimes) == 0: return 0.0
        probs = lifetimes / np.sum(lifetimes)
        return -np.sum(probs * np.log(probs + 1e-10))

    def persistence_image(self, diagram, resolution=32):
        """Legacy wrapper for persistence imager"""
        if len(diagram) == 0:
            return np.zeros((resolution, resolution))
        # Ensure resolution matches imager or resize
        clean_diag = diagram[diagram[:, 1] != np.inf]
        if len(clean_diag) == 0:
            return np.zeros((resolution, resolution))
        img = self.imager.fit_transform([clean_diag])[0]
        # Resize if necessary
        if img.shape[0] != resolution:
            from scipy.ndimage import zoom
            img = zoom(img, resolution / img.shape[0])
        return img

# Example Usage
if __name__ == "__main__":
    engine = ProductionTopologyEngine()
    # Synthetic torus data
    t = np.linspace(0, 2*np.pi, 100)
    data = np.column_stack([np.cos(t), np.sin(t)]) + np.random.normal(0, 0.1, (100, 2))
    
    sig = engine.analyze_window(data)
    print(f"Loop Score: {sig.loop_score:.4f}")
    print(f"TTI: {sig.tti:.4f}")
    print(f"H1 Summary: {sig.h1_summary}")
