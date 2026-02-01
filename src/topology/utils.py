import numpy as np
import torch
from scipy.stats import norm

def persistence_diagram_to_image(persistence_points: np.ndarray, 
                                resolution: int = 32, 
                                sigma: float = 0.05,
                                max_val: float = 1.0) -> np.ndarray:
    """
    Converts a persistence diagram [(birth, death), ...] to a persistence image.
    Standard financial TDA technique used by JPM/RenTech.
    """
    if len(persistence_points) == 0:
        return np.zeros((1, resolution, resolution), dtype=np.float32)

    # 1. Map to Persistence Space (birth, persistence)
    # persistence = death - birth
    pts = np.copy(persistence_points)
    pts[:, 1] = pts[:, 1] - pts[:, 0]
    
    # 2. Filter infinite or invalid points
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return np.zeros((1, resolution, resolution), dtype=np.float32)

    # 3. Create Grid
    # Normalize birth [0, max_val] and persistence [0, max_val]
    grid = np.zeros((resolution, resolution), dtype=np.float32)
    
    # Simple histogram binning with Gaussian smoothing
    # Note: In production, we'd use a faster kernel density estimate
    # Here we do a localized Gaussian contribution for each point
    
    x_bins = np.linspace(0, max_val, resolution)
    y_bins = np.linspace(0, max_val, resolution)
    
    for (birth, persist) in pts:
        # Bin indices
        ix = np.argmin(np.abs(x_bins - birth))
        iy = np.argmin(np.abs(y_bins - persist))
        
        # Local Gaussian influence (kernel)
        # We'll just increment the bin for now for speed, 
        # but weighting by persistence is standard (larger holes = more signal)
        grid[iy, ix] += persist
        
    # Apply Gaussian Blur (Optional but recommended for stability)
    from scipy.ndimage import gaussian_filter
    grid = gaussian_filter(grid, sigma=sigma * resolution)
    
    # Normalize
    if np.max(grid) > 0:
        grid = grid / np.max(grid)
        
    return grid.reshape(1, resolution, resolution)

def get_market_stream_features(l3_book: dict, recent_trades: list) -> np.ndarray:
    """
    Extracts the 7-dim feature vector for NeuralCDE:
    [price, bid_vol, ask_vol, trade_buy, trade_sell, canc_buy, canc_sell]
    """
    # 1. Mid Price
    bids = np.array(l3_book.get('bids', [[0,0]]))
    asks = np.array(l3_book.get('asks', [[0,0]]))
    mid = (bids[0, 0] + asks[0, 0]) / 2.0
    
    # 2. Volumes
    bid_vol = np.sum(bids[:, 1])
    ask_vol = np.sum(asks[:, 1])
    
    # 3. Trades
    buy_trade_vol = sum(t['size'] for t in recent_trades if t.get('side') == 'buy')
    sell_trade_vol = sum(t['size'] for t in recent_trades if t.get('side') == 'sell')
    
    # 4. Cancellations (Simplified proxy for now)
    # Usually calculated by tracking L3 deltas
    canc_buy = 0.0 # To be populated by state tracking
    canc_sell = 0.0
    
    return np.array([mid, bid_vol, ask_vol, buy_trade_vol, sell_trade_vol, canc_buy, canc_sell], dtype=np.float32)
