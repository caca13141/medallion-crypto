"""
Bifiltrated Persistence: Correlation × Volume × Volatility
Multi-parameter persistent homology for market microstructure
"""
import numpy as np
import gudhi
from scipy.spatial.distance import pdist, squareform

class BifiltrationEngine:
    """
    Implements bifiltrated persistence:
    - Primary filtration: Vietoris-Rips (distance/correlation)
    - Secondary filtration: Volume × Volatility weight
    
    This captures both geometric (price correlation) and economic (volume/vol) structure.
    """
    
    def __init__(self, max_dimension=2):
        self.max_dimension = max_dimension
        
    def compute_correlation_matrix(self, returns_matrix):
        """
        Compute pairwise correlation between time windows.
        Input: (n_windows, window_size) array of returns
        Output: (n_windows, n_windows) correlation matrix
        """
        # Pearson correlation
        correlations = np.corrcoef(returns_matrix)
        # Convert to distance: d = sqrt(2(1 - corr))
        distances = np.sqrt(2 * (1 - correlations))
        return distances
    
    def compute_weighted_distance_matrix(self, point_cloud, volumes, volatilities):
        """
        Compute distance matrix with volume × volatility weighting.
        High volume × high vol regions get smaller effective distances (more connectedness).
        """
        # Euclidean distances
        distances = squareform(pdist(point_cloud, metric='euclidean'))
        
        # Volume × Volatility weights (normalize)
        weights = volumes * volatilities
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
        
        # Weighted distance: d_ij = d_ij / sqrt(w_i * w_j)
        # This makes high-activity regions "closer"
        weight_matrix = np.outer(weights, weights)
        weighted_distances = distances / (np.sqrt(weight_matrix) + 1e-8)
        
        return weighted_distances
    
    def compute_bifiltrated_persistence(self, returns_windows, volumes, volatilities):
        """
        Main bifiltration computation.
        
        Args:
            returns_windows: (n_windows, window_size) - sliding windows of returns
            volumes: (n_windows,) - volume for each window
            volatilities: (n_windows,) - realized volatility for each window
            
        Returns:
            Dictionary with:
            - 'corr_diagrams': persistence diagrams from correlation filtration
            - 'weighted_diagrams': persistence diagrams from weighted filtration
            - 'zigzag_features': zigzag persistence features (birth/death of features)
        """
        # Correlation-based filtration
        corr_distances = self.compute_correlation_matrix(returns_windows)
        corr_diagrams = self._compute_vr_persistence(corr_distances)
        
        # Volume × Volatility weighted filtration
        # First create point cloud from returns (Takens embedding)
        point_cloud = self._create_point_cloud(returns_windows)
        weighted_distances = self.compute_weighted_distance_matrix(
            point_cloud, volumes, volatilities
        )
        weighted_diagrams = self._compute_vr_persistence(weighted_distances)
        
        # Zigzag persistence (track feature birth/death across both filtrations)
        zigzag_features = self._compute_zigzag_features(
            corr_diagrams, weighted_diagrams
        )
        
        return {
            'corr_diagrams': corr_diagrams,
            'weighted_diagrams': weighted_diagrams,
            'zigzag_features': zigzag_features
        }
    
    def _create_point_cloud(self, returns_windows):
        """Simple 3D Takens embedding for each window"""
        n_windows = len(returns_windows)
        cloud = np.zeros((n_windows, 3))
        
        for i, window in enumerate(returns_windows):
            if len(window) >= 3:
                cloud[i, 0] = window[-1]
                cloud[i, 1] = window[-2] if len(window) > 1 else 0
                cloud[i, 2] = window[-3] if len(window) > 2 else 0
                
        return cloud
    
    def _compute_vr_persistence(self, distance_matrix):
        """
        Compute Vietoris-Rips persistence from distance matrix using GUDHI.
        """
        rips = gudhi.RipsComplex(distance_matrix=distance_matrix, max_edge_length=100)
        simplex_tree = rips.create_simplex_tree(max_dimension=self.max_dimension)
        persistence = simplex_tree.persistence()
        
        # Convert to numpy arrays by dimension
        diagrams = {i: [] for i in range(self.max_dimension + 1)}
        for dim, (birth, death) in persistence:
            if death != float('inf'):
                diagrams[dim].append([birth, death])
                
        for dim in diagrams:
            diagrams[dim] = np.array(diagrams[dim]) if diagrams[dim] else np.zeros((0, 2))
            
        return diagrams
    
    def _compute_zigzag_features(self, diagrams1, diagrams2):
        """
        Zigzag persistence: track how features appear/disappear across filtrations.
        
        Returns:
            - num_births_corr: features born in correlation but not in weighted
            - num_births_weighted: features born in weighted but not in correlation
            - num_deaths_corr: features dying earlier in correlation
            - persistence_ratio: ratio of total persistence between filtrations
        """
        features = {}
        
        for dim in [0, 1]:  # H0 and H1
            d1 = diagrams1.get(dim, np.zeros((0, 2)))
            d2 = diagrams2.get(dim, np.zeros((0, 2)))
            
            # Count features
            n1 = len(d1)
            n2 = len(d2)
            
            # Total persistence
            pers1 = np.sum(d1[:, 1] - d1[:, 0]) if n1 > 0 else 0
            pers2 = np.sum(d2[:, 1] - d2[:, 0]) if n2 > 0 else 0
            
            features[f'H{dim}_births_corr'] = n1
            features[f'H{dim}_births_weighted'] = n2
            features[f'H{dim}_persistence_corr'] = pers1
            features[f'H{dim}_persistence_weighted'] = pers2
            features[f'H{dim}_persistence_ratio'] = pers1 / (pers2 + 1e-8)
            
        return features
    
    def extract_bifiltration_features(self, bifiltration_result):
        """
        Extract quantitative features from bifiltration for ML models.
        
        Returns a flat feature vector.
        """
        features = []
        
        # Correlation-based features
        corr_diag = bifiltration_result['corr_diagrams']
        for dim in [0, 1]:
            d = corr_diag.get(dim, np.zeros((0, 2)))
            if len(d) > 0:
                lifetimes = d[:, 1] - d[:, 0]
                features.extend([
                    len(d),  # number of features
                    np.mean(lifetimes),  # average persistence
                    np.max(lifetimes),  # maximum persistence
                    np.std(lifetimes),  # persistence variation
                ])
            else:
                features.extend([0, 0, 0, 0])
        
        # Weighted-based features
        weighted_diag = bifiltration_result['weighted_diagrams']
        for dim in [0, 1]:
            d = weighted_diag.get(dim, np.zeros((0, 2)))
            if len(d) > 0:
                lifetimes = d[:, 1] - d[:, 0]
                features.extend([
                    len(d),
                    np.mean(lifetimes),
                    np.max(lifetimes),
                    np.std(lifetimes),
                ])
            else:
                features.extend([0, 0, 0, 0])
        
        # Zigzag features
        zigzag = bifiltration_result['zigzag_features']
        features.extend([
            zigzag.get('H0_persistence_ratio', 0),
            zigzag.get('H1_persistence_ratio', 0),
            zigzag.get('H0_births_corr', 0) - zigzag.get('H0_births_weighted', 0),
            zigzag.get('H1_births_corr', 0) - zigzag.get('H1_births_weighted', 0),
        ])
        
        return np.array(features)
