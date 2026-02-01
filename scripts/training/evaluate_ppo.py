"""
Evaluate Trained PPO Agent
Tests the trained policy on held-out data and generates performance report.
"""
import sys
sys.path.append('/Users/raphaelmaksoud/crypto toppo')

from src.rl.continuous_ppo import ContinuousTopoEnv, ProductionAgent
import numpy as np
from stable_baselines3 import PPO

print("=" * 60)
print(" PPO Agent Evaluation")
print("=" * 60)

# Load trained model
model_path = "src/data/ppo_final.zip"
print(f"\n Loading trained model: {model_path}")

env = ContinuousTopoEnv()
model = PPO.load(model_path, env=env)

print(f" Model loaded successfully")
print(f"   Total samples in dataset: {len(env.data)}")
print(f"   Training used first ~80%")
print(f"\n Testing on held-out last 20%...")

# Test on last 20% of data
test_start = int(len(env.data) * 0.8)
num_episodes = 10
results = []

for episode in range(num_episodes):
    obs, _ = env.reset()
    # Force reset to test set
    env.current_step = test_start + (episode * 100)
    
    episode_reward = 0
    episode_length = 0
    actions_taken = []
    
    for _ in range(200):  # Max 200 steps per episode
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        episode_reward += reward
        episode_length += 1
        actions_taken.append(action)
        
        if done or truncated:
            break
    
    final_equity = env.equity
    pnl_pct = (final_equity - 10000) / 10000 * 100
    
    results.append({
        'episode': episode + 1,
        'reward': episode_reward,
        'length': episode_length,
        'final_equity': final_equity,
        'pnl_pct': pnl_pct,
        'avg_leverage': np.mean([3.0 + a[0] * 27.0 for a in actions_taken]),
        'avg_position': np.mean([a[1] for a in actions_taken])
    })
    
    print(f"Episode {episode+1:2d}: Reward={episode_reward:7.2f} | Equity=${final_equity:,.2f} | PnL={pnl_pct:+6.2f}%")

# Summary statistics
print("\n" + "=" * 60)
print(" Performance Summary")
print("=" * 60)

avg_reward = np.mean([r['reward'] for r in results])
avg_pnl = np.mean([r['pnl_pct'] for r in results])
win_rate = sum(1 for r in results if r['pnl_pct'] > 0) / len(results)
avg_leverage = np.mean([r['avg_leverage'] for r in results])
avg_position = np.mean([r['avg_position'] for r in results])

print(f"\n Returns:")
print(f"   Average PnL: {avg_pnl:+.2f}%")
print(f"   Win Rate: {win_rate*100:.1f}%")
print(f"   Best Episode: {max(r['pnl_pct'] for r in results):+.2f}%")
print(f"   Worst Episode: {min(r['pnl_pct'] for r in results):+.2f}%")

print(f"\n Strategy:")
print(f"   Avg Leverage: {avg_leverage:.1f}x")
print(f"   Avg Position Size: {avg_position*100:.1f}%")
print(f"   Avg Reward per Episode: {avg_reward:.2f}")

print(f"\n Comparison:")
print(f"   Trained Agent PnL: {avg_pnl:+.2f}%")
print(f"   Buy & Hold (baseline): ~0% (flat market)")

print("\n" + "=" * 60)
if avg_pnl > 0:
    print(" Agent shows positive expected returns!")
else:
    print("  Agent needs more training or hyperparameter tuning")
print("=" * 60)
