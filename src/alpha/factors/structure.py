import numpy as np

def calculate_zigzag(df, threshold=0.02):
    prices = df['c'].values
    pivots = np.zeros(len(prices))
    
    last_pivot_price = prices[0]
    last_pivot_idx = 0
    trend = 0 
    
    for i in range(1, len(prices)):
        curr = prices[i]
        dev = (curr - last_pivot_price) / last_pivot_price
        
        if trend == 0:
            if dev > threshold: trend = 1; last_pivot_price = curr; last_pivot_idx = i
            elif dev < -threshold: trend = -1; last_pivot_price = curr; last_pivot_idx = i
        elif trend == 1:
            if curr > last_pivot_price:
                last_pivot_price = curr
                last_pivot_idx = i
            elif dev < -threshold:
                pivots[last_pivot_idx] = 1 
                trend = -1
                last_pivot_price = curr
                last_pivot_idx = i
        elif trend == -1:
            if curr < last_pivot_price:
                last_pivot_price = curr
                last_pivot_idx = i
            elif dev > threshold:
                pivots[last_pivot_idx] = -1 
                trend = 1
                last_pivot_price = curr
                last_pivot_idx = i
                
    return pivots
