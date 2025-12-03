"""
Persistent Homology Core Engine
Replaces Hurst exponent with real topological analysis
"""
import numpy as np
import gudhi
from ripser import ripser
from persim import wasserstein
import warnings
warnings.filterwarnings('ignore')

class PersistenceEngine:
    """
    Core persistence homology computation engine.
    Generates persistence diagrams, barcodes, and topological features.
    """
    
    def __init__(self, max_dimension=2, max_edge_length=100.0):
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
        
    def compute_point_cloud(self, prices, volumes=None, volatilities=None):
        """
        Create 3D point cloud from time series data:
        - x: normalized price returns
        - y: volume (if available)
        - z: volatility (if available)
        """
        returns = np.diff(np.log(prices))
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        if volumes is not None and volatilities is not None:
            # 3D cloud
            vols = (volumes - volumes.mean()) / (volumes.std() + 1e-8)
            volas = (volatilities - volatilities.mean()) / (volatilities.std() + 1e-8)
            cloud = np.column_stack([returns, vols[:-1], volas[:-1]])
        elif volumes is not None:
            # 2D cloud
            vols = (volumes - volumes.mean()) / (volumes.std() + 1e-8)
            cloud = np.column_stack([returns, vols[:-1]])
        else:
            # 1D embedding via Takens
            cloud = self._takens_embed(returns, dim=3, delay=1)
            
        return cloud
    
    def _takens_embed(self, series, dim=3, delay=1):
        """Takens delay embedding for univariate series"""
        n = len(series) - (dim - 1) * delay
        embedded = np.zeros((n, dim))
        for i in range(dim):
            embedded[:, i] = series[i * delay : i * delay + n]
        return embedded
    
    def compute_persistence(self, point_cloud):
        """
        Compute persistence diagram using Ripser (fast Vietoris-Rips).
        Returns diagrams for H0 (connected components) and H1 (loops).
        """
        result = ripser(point_cloud, maxdim=self.max_dimension, thresh=self.max_edge_length)
        diagrams = result['dgms']
        return diagrams
    
    def compute_barcodes(self, point_cloud):
        """
        Compute signed persistence barcodes using GUDHI.
        Returns birth-death pairs for each homology dimension.
        """
        rips = gudhi.RipsComplex(points=point_cloud, max_edge_length=self.max_edge_length)
        simplex_tree = rips.create_simplex_tree(max_dimension=self.max_dimension)
        persistence = simplex_tree.persistence()
        
        barcodes = {i: [] for i in range(self.max_dimension + 1)}
        for dim, (birth, death) in persistence:
            if death != float('inf'):
                barcodes[dim].append((birth, death))
                
        return barcodes
    
    def persistence_entropy(self, diagram):
        """
        Calculate persistence entropy as a measure of topological complexity.
        High entropy = turbulent/noisy market.
        Low entropy = clear structure.
        """
        if len(diagram) == 0:
            return 0.0
            
        lifetimes = diagram[:, 1] - diagram[:, 0]
        lifetimes = lifetimes[np.isfinite(lifetimes)]
        
        if len(lifetimes) == 0:
            return 0.0
            
        L = lifetimes.sum()
        if L == 0:
            return 0.0
            
        p = lifetimes / L
        entropy = -np.sum(p * np.log(p + 1e-10))
        return entropy
    
    def loop_score(self, diagrams):
        """
        Calculate Loop Score: strength and persistence of H1 loops.
        High score = strong cyclical patterns (mean reversion signal).
        Low score = trending market.
        """
        if len(diagrams) < 2:
            return 0.0
            
        h1_diagram = diagrams[1]  # H1 loops
        if len(h1_diagram) == 0:
            return 0.0
            
        # Filter infinite points
        h1_finite = h1_diagram[h1_diagram[:, 1] != np.inf]
        if len(h1_finite) == 0:
            return 0.0
            
        lifetimes = h1_finite[:, 1] - h1_finite[:, 0]
        
        # Loop score = sum of squared lifetimes (emphasizes persistent loops)
        score = np.sum(lifetimes ** 2)
        return score
    
    def topological_turbulence_index(self, diagrams):
        """
        TTI = Persistence Entropy (H0) + 2 * Persistence Entropy (H1)
        Higher values = dangerous volatility regime
        """
        h0_ent = self.persistence_entropy(diagrams[0]) if len(diagrams) > 0 else 0.0
        h1_ent = self.persistence_entropy(diagrams[1]) if len(diagrams) > 1 else 0.0
        tti = h0_ent + 2.0 * h1_ent
        return tti
    
    def persistence_image(self, diagram, resolution=20, sigma=0.1):
        """
        Convert persistence diagram to persistence image (for ML input).
        Returns a 2D image representation of the diagram.
        """
        if len(diagram) == 0:
            return np.zeros((resolution, resolution))
            
        # Filter infinite points
        finite_diagram = diagram[diagram[:, 1] != np.inf]
        if len(finite_diagram) == 0:
            return np.zeros((resolution, resolution))
            
        # Create grid
        birth_min, birth_max = finite_diagram[:, 0].min(), finite_diagram[:, 0].max()
        death_min, death_max = finite_diagram[:, 1].min(), finite_diagram[:, 1].max()
        
        if birth_max == birth_min:
            birth_max = birth_min + 1
        if death_max == death_min:
            death_max = death_min + 1
            
        x_bins = np.linspace(birth_min, birth_max, resolution)
        y_bins = np.linspace(death_min, death_max, resolution)
        
        # Gaussian smoothing
        image = np.zeros((resolution, resolution))
        for birth, death in finite_diagram:
            lifetime = death - birth
            x_idx = np.searchsorted(x_bins, birth)
            y_idx = np.searchsorted(y_bins, death)
            
            if 0 <= x_idx < resolution and 0 <= y_idx < resolution:
                # Gaussian kernel weighted by persistence
                for i in range(max(0, x_idx - 2), min(resolution, x_idx + 3)):
                    for j in range(max(0, y_idx - 2), min(resolution, y_idx + 3)):
                        dist = np.sqrt((x_bins[i] - birth)**2 + (y_bins[j] - death)**2)
                        weight = lifetime * np.exp(-dist**2 / (2 * sigma**2))
                        image[i, j] += weight
                        
        return image
