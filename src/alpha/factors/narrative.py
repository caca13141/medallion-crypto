import kmapper as km
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from src.config import Config

class NarrativeMapper:
    def __init__(self):
        self.mapper = km.KeplerMapper(verbose=0)
        self.projector = PCA(n_components=2)
        self.clusterer = DBSCAN(eps=0.5, min_samples=3)
        
    def map_narrative(self, df):
        # Needs historical context to map
        if len(df) < 50: return -1
        
        # Features for TDA
        features = df[['c', 'o', 'h', 'l', 'v']].values
        # Normalize
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        
        # Project
        projected = self.mapper.project(
            features, 
            projection=self.projector
        )
        
        # Cluster (Map)
        # We map the ENTIRE sequence to find the structure, 
        # but we only care about the cluster ID of the LAST point (current state).
        
        # KeplerMapper usually builds a graph. For real-time narrative ID, 
        # we can simplify: Project -> Cluster directly in the projected space 
        # OR use the graph nodes.
        # For speed/MVP: Direct clustering on projection of recent window.
        
        # Fit clusterer on projected data
        labels = self.clusterer.fit_predict(projected)
        
        # Return the label of the last data point
        return labels[-1]
