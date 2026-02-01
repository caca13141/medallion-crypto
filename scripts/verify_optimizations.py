import time
import numpy as np
import torch
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("."))

from src.topology.persistence_core import ProductionTopologyEngine
from src.signals.hawkes_cascade import HawkesCascadeEngine
from src.signals.rough_path_signature import SignatureModel
from src.control.deep_signature_pde import DeepSignaturePDE
from src.rl.continuous_ppo import ContinuousTopoEnv
from src.risk.risk_system import HierarchicalRiskParity

def test_topology():
    print("\n--- Testing Topology Optimization ---")
    engine = ProductionTopologyEngine()
    # Create large point cloud
    pc = np.random.randn(500, 4) # 500 points would normally slow down Ripser
    t0 = time.time()
    sig = engine.analyze_window(pc)
    dt = (time.time() - t0) * 1000
    print(f"Analysis of 500 points took {dt:.2f} ms")
    assert dt < 400, "Topology analysis too slow!"
    print(" Topology Memory/Speed Check Passed")

def test_hawkes():
    print("\n--- Testing Hawkes Optimization ---")
    engine = HawkesCascadeEngine()
    # Warmup
    for i in range(100):
        engine.add_event(np.random.randint(0, 4), i * 0.01)
    
    t0 = time.time()
    n_events = 10000
    for i in range(n_events):
        engine.add_event(np.random.randint(0, 4), 100 + i * 0.001)
        _ = engine.get_hawkes_score(100 + i * 0.001)
        
    dt = time.time() - t0
    rate = n_events / dt
    print(f"Processed {n_events} events in {dt:.4f}s. Rate: {rate:.0f} events/sec")
    assert rate > 5000, "Hawkes throughput too low!"
    print(" Hawkes Speed Check Passed")

def test_signature():
    print("\n--- Testing Signature Optimization ---")
    model = SignatureModel(depth=4, output_dim=128)
    x = torch.randn(16, 50, 4)
    loop_score, leverage = model(x)
    print(f"Signature Model Output: Loop={loop_score.shape}, Lev={leverage.shape}")
    print(" Signature Model Check Passed")

def test_pde():
    print("\n--- Testing Deep PDE Latency ---")
    solver = DeepSignaturePDE(device="cpu")
    path = np.random.randn(50, 4).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        solver.solve(path)
        
    times = []
    for _ in range(100):
        t0 = time.time()
        solver.solve(path)
        times.append((time.time() - t0) * 1000)
        
    avg_time = np.mean(times)
    print(f"Average PDE Solve Time: {avg_time:.2f} ms")
    assert avg_time < 5.0, f"PDE Solver too slow! ({avg_time:.2f} ms)"
    print(" Deep PDE Latency Check Passed")

def test_ppo():
    print("\n--- Testing PPO Diffusion Augmentation ---")
    env = ContinuousTopoEnv(augment=True)
    obs, _ = env.reset()
    # Step a few times
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
    print(" PPO Augmentation Check Passed")

def test_risk():
    print("\n--- Testing Risk HRP ---")
    # Generate fake returns
    returns = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
    # Add correlation
    returns['B'] = returns['A'] * 0.9 + np.random.randn(100) * 0.1
    
    weights = HierarchicalRiskParity.allocate(returns)
    print("HRP Weights:")
    print(weights)
    assert np.isclose(weights.sum(), 1.0), "Weights must sum to 1"
    print(" Risk HRP Check Passed")

if __name__ == "__main__":
    test_topology()
    test_hawkes()
    test_signature()
    test_pde()
    test_ppo()
    test_risk()
    print("\n ALL CHECKS PASSED ")
