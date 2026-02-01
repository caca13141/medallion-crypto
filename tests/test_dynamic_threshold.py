"""
Test Dynamic TTI Threshold
Verifies z-score based adaptive threshold works correctly.
"""
import sys
sys.path.append('/Users/raphaelmaksoud/crypto toppo')

from src.risk.risk_system import RiskManagementSystem
import numpy as np

print("=" * 60)
print(" Testing Dynamic TTI Threshold")
print("=" * 60)

# Initialize risk system
risk = RiskManagementSystem(tti_threshold=8.0)

print(f"\n Configuration:")
print(f"   Static threshold: {risk.tti_threshold}")
print(f"   Z-score threshold: {risk.tti_z_threshold}")
print(f"   Warmup samples needed: {risk.min_samples_for_zscore}")
print(f"   Dynamic mode enabled: {risk.use_dynamic_threshold}")

# Simulate stable market (TTI around 8.0-8.2)
print(f"\n Simulating Stable Market (50 iterations)...")
print(f"   TTI values: 8.0-8.2 (slightly above old threshold)")

for i in range(60):
    # Stable TTI around 8.1
    tti = 8.0 + np.random.uniform(0, 0.2)
    result = risk.check_risk({'tti': tti}, model_confidence=0.8)
    
    if i < 50:
        # Warmup mode
        if i == 0:
            print(f"\n   Iteration {i}: TTI={tti:.2f} - {result.reason}")
        elif i == 49:
            print(f"   Iteration {i}: TTI={tti:.2f} - {result.reason}")
    else:
        # Z-score mode
        if i == 50:
            tti_mean = np.mean(risk.tti_history)
            tti_std = np.std(risk.tti_history)
            print(f"\n    Switched to z-score mode at iteration {i}")
            print(f"      Baseline: μ={tti_mean:.2f}, σ={tti_std:.2f}")
        
        print(f"   Iteration {i}: TTI={tti:.2f} - {result.reason} - Can trade: {result.can_trade}")

# Test spike detection
print(f"\n Simulating TTI Spike...")
spike_tti = 10.0  # Way above normal
result = risk.check_risk({'tti': spike_tti}, model_confidence=0.8)

tti_mean = np.mean(risk.tti_history)
tti_std = np.std(risk.tti_history)
z_score = (spike_tti - tti_mean) / tti_std

print(f"   Spike TTI: {spike_tti:.2f}")
print(f"   Calculated z-score: {z_score:.2f}")
print(f"   Result: {result.reason}")
print(f"   Kill-switch active: {not result.can_trade}")

print("\n" + "=" * 60)
print(" Test Complete!")
print("=" * 60)
