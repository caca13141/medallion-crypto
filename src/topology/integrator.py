"""
Topology Integrator: Connects persistence engine to signal pipeline
Replaces Hurst exponent with full topological analysis
"""
import numpy as np
import pandas as pd
from src.topology.persistence_core import PersistenceEngine
from src.topology.bifiltration import BifiltrationEngine

class TopologyIntegrator:
    """
    Integrates persistent homology into the trading signal pipeline.
    
    Replaces:
    - Hurst Exponent → Loop Score + TTI (Topological Turbulence Index)
    - Single metric → Rich topological feature vector
    
    Outputs:
    - loop_score: strength of H1 cycles (mean reversion signal)
    - tti: topological turbulence (risk-off trigger)
    - persistence_images: for Transformer input
    - bifiltration_features: for ML models
    """
    
    def __init__(self, lookback=100, resolution=20):
        self.lookback = lookback
        self.resolution = resolution
        
        self.persistence_engine = PersistenceEngine(max_dimension=2)
        self.bifiltration_engine = BifiltrationEngine(max_dimension=2)
        
    def analyze(self, df):
        """
        Main analysis function.
        
        Args:
            df: DataFrame with columns ['c', 'h', 'l', 'v'] (close, high, low, volume)
            
        Returns:
            dict with:
            - loop_score: float
            - tti: float
            - persistence_images_h0: (resolution, resolution)
            - persistence_images_h1: (resolution, resolution)
            - bifiltration_features: (n_features,)
            - diagrams: raw persistence diagrams
        """
        if len(df) < self.lookback:
            return self._empty_result()
            
        # Take last N candles
        recent = df.tail(self.lookback).copy()
        
        # Extract data
        prices = recent['c'].values
        volumes = recent['v'].values if 'v' in recent.columns else None
        
        # Calculate volatility (high-low range)
        if 'h' in recent.columns and 'l' in recent.columns:
            volatilities = (recent['h'].values - recent['l'].values) / prices
        else:
            # Fallback: use returns volatility
            returns = np.diff(np.log(prices))
            volatilities = np.abs(returns)
            volatilities = np.append(volatilities, volatilities[-1])
        
        # 1. Basic persistence analysis
        point_cloud = self.persistence_engine.compute_point_cloud(
            prices, volumes, volatilities
        )
        diagrams = self.persistence_engine.compute_persistence(point_cloud)
        
        # 2. Core metrics
        loop_score = self.persistence_engine.loop_score(diagrams)
        tti = self.persistence_engine.topological_turbulence_index(diagrams)
        
        # 3. Persistence images (for Transformer)
        h0_image = self.persistence_engine.persistence_image(
            diagrams[0], resolution=self.resolution
        )
        h1_image = self.persistence_engine.persistence_image(
            diagrams[1], resolution=self.resolution
        ) if len(diagrams) > 1 else np.zeros((self.resolution, self.resolution))
        
        # 4. Bifiltration analysis (if volume/vol available)
        bifiltration_features = np.zeros(20)  # Default
        if volumes is not None:
            try:
                # Create sliding windows for bifiltration
                window_size = 10
                n_windows = len(prices) - window_size + 1
                returns = np.diff(np.log(prices))
                
                returns_windows = np.array([
                    returns[i:i+window_size] 
                    for i in range(n_windows)
                ])
                
                vol_windows = np.array([
                    np.mean(volumes[i:i+window_size])
                    for i in range(n_windows)
                ])
                
                vola_windows = np.array([
                    np.mean(volatilities[i:i+window_size])
                    for i in range(n_windows)
                ])
                
                bifiltration = self.bifiltration_engine.compute_bifiltrated_persistence(
                    returns_windows, vol_windows, vola_windows
                )
                
                bifiltration_features = self.bifiltration_engine.extract_bifiltration_features(
                    bifiltration
                )
            except Exception as e:
                # Bifiltration can fail on small/noisy data
                pass
        
        return {
            'loop_score': loop_score,
            'tti': tti,
            'persistence_image_h0': h0_image,
            'persistence_image_h1': h1_image,
            'bifiltration_features': bifiltration_features,
            'diagrams': diagrams,
            'point_cloud': point_cloud
        }
    
    def _empty_result(self):
        """Return empty result when data insufficient"""
        return {
            'loop_score': 0.0,
            'tti': 0.0,
            'persistence_image_h0': np.zeros((self.resolution, self.resolution)),
            'persistence_image_h1': np.zeros((self.resolution, self.resolution)),
            'bifiltration_features': np.zeros(20),
            'diagrams': [np.zeros((0, 2)), np.zeros((0, 2))],
            'point_cloud': np.zeros((0, 3))
        }
    
    def should_flatten(self, tti, threshold=3.0):
        """
        Kill-switch based on topological turbulence.
        
        Args:
            tti: topological turbulence index
            threshold: TTI threshold for flattening
            
        Returns:
            True if should flatten all positions (danger)
        """
        return tti > threshold
    
    def get_regime(self, loop_score, tti):
        """
        Classify market regime based on topology.
        
        Returns:
            'trending': low loops, low turbulence → momentum
            'mean_reverting': high loops, low turbulence → range-bound
            'volatile': high turbulence → dangerous
        """
        if tti > 2.5:
            return 'volatile'
        elif loop_score > 0.5:
            return 'mean_reverting'
        else:
            return 'trending'
