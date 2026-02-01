import numpy as np
try:
    import gudhi
except ImportError:
    gudhi = None

class OrderBookTopology:
    """
    Phase 2: Order-Book-Aware TDA (TTI 2.0).
    Predicts microstructure breakdowns and liquidations.
    """
    def __init__(self, max_edge_length=0.01):
        self.max_edge_length = max_edge_length

    def compute_microstructure_tti(self, 
                                 bids: np.ndarray, 
                                 asks: np.ndarray, 
                                 cancellations: float) -> float:
        """
        Computes Topological Turbulence Index (TTI) from L3 Book.
        
        Args:
            bids: (50, 2) array [price, size]
            asks: (50, 2) array [price, size]
            cancellations: float (normalized count last 5s)
            
        Returns:
            tti_micro: float (Predictive signal for volatility/liquidation)
        """
        if gudhi is None:
            return 0.0
            
        # 1. Construct 4D Point Cloud
        # Dimensions: [Price, Size, Side(-1/1), Depth_Rank]
        
        # Normalize Prices (local z-score-ish)
        mid_price = (bids[0, 0] + asks[0, 0]) / 2.0
        price_scale = (bids[0, 0] - bids[-1, 0]) + 1e-6
        
        bid_pts = np.zeros((len(bids), 4))
        bid_pts[:, 0] = (bids[:, 0] - mid_price) / price_scale
        bid_pts[:, 1] = np.log1p(bids[:, 1]) # Log size
        bid_pts[:, 2] = -1.0 # Side
        bid_pts[:, 3] = np.arange(len(bids)) / 50.0 # Rank
        
        ask_pts = np.zeros((len(asks), 4))
        ask_pts[:, 0] = (asks[:, 0] - mid_price) / price_scale
        ask_pts[:, 1] = np.log1p(asks[:, 1])
        ask_pts[:, 2] = 1.0 # Side
        ask_pts[:, 3] = np.arange(len(asks)) / 50.0 # Rank
        
        # Combine
        point_cloud = np.vstack([bid_pts, ask_pts])
        
        # 2. Compute Persistence (Alpha Complex or Rips)
        # Alpha complex is faster for low-dim, but Rips is standard
        rips = gudhi.RipsComplex(points=point_cloud, max_edge_length=self.max_edge_length)
        st = rips.create_simplex_tree(max_dimension=2)
        st.persistence()
        
        # 3. Extract Features
        h0 = st.persistence_intervals_in_dimension(0)
        h1 = st.persistence_intervals_in_dimension(1)
        
        # TTI Calculation: Entropy of H0 + Sum of H1 lifetimes
        # Weighted by cancellation intensity
        
        # H0 Entropy (Fragmentation of liquidity)
        h0_lens = h0[:, 1] - h0[:, 0]
        h0_lens = h0_lens[np.isfinite(h0_lens)]
        if len(h0_lens) > 0:
            probs = h0_lens / np.sum(h0_lens)
            h0_entropy = -np.sum(probs * np.log(probs + 1e-10))
        else:
            h0_entropy = 0.0
            
        # H1 Persistence (Loops/Holes in liquidity)
        h1_lens = h1[:, 1] - h1[:, 0]
        h1_sum = np.sum(h1_lens) if len(h1_lens) > 0 else 0.0
        
        # Microstructure TTI
        # High Entropy (Fragmented) + High H1 (Gaps) + High Cancellations = CRASH
        tti_micro = (h0_entropy + h1_sum) * (1.0 + cancellations)
        
        return float(tti_micro)
