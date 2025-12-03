import numpy as np
import pandas as pd

def calculate_hurst(series, min_window=10):
    """
    Calculate the Hurst Exponent of a time series.
    H < 0.5: Mean Reverting (Anti-persistent)
    H ~ 0.5: Random Walk (Brownian Motion)
    H > 0.5: Trending (Persistent)
    """
    if len(series) < 100: return 0.5 # Not enough data
    
    # Simplified R/S Analysis
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    
    # Use polyfit to estimate H
    # log(tau) = H * log(lag) + C
    # This is actually for Generalized Hurst or similar scaling. 
    # Standard R/S is complex to implement robustly in few lines.
    # Using Variance Ratio test proxy or simplified Hurst.
    
    # Let's use a robust simplified version:
    # H = log(R/S) / log(N)
    
    # Actually, let's use the standard library approach if we had 'hurst' package, 
    # but we'll implement a basic one here.
    
    # Vectorized Rescaled Range (R/S) Analysis
    series = np.array(series)
    N = len(series)
    
    # Split into chunks
    # For MVP, we calculate over the whole window
    mean = np.mean(series)
    deviations = series - mean
    cum_deviations = np.cumsum(deviations)
    R = np.max(cum_deviations) - np.min(cum_deviations)
    S = np.std(series)
    
    if S == 0: return 0.5
    
    RS = R / S
    H = np.log(RS) / np.log(N)
    
    return H
