"""
Evaluate Aggressive PPO Agent
Tests the aggressive policy on held-out data.
"""
import sys
sys.path.append('/Users/raphaelmaksoud/crypto toppo')

from train_aggressive_ppo import AggressiveTopoEnv
import numpy as np
from stable_baselines3 import PPO

print("=" * 60)
print(" Aggressive PPO Agent Evaluation")
print("=" * 60)

# Load aggressive model
model_path = "src/data/ppo_aggressive.zip"
print(f"\n Loading aggressive model: {model_path}")

env = AggressiveTopoEnv()
model = PPO.load(model_path, env=env)

print(f" Model loaded successfully")
print(f"   Total samples: {len(env.data)}")
print(f"\n Testing on held-out last 20%...")

# Test on last 20%
test_start = int(len(env.data) * 0.8)
num_episodes = 10
results = []

for episode in range(num_episodes):
    obs, _ = env.reset()
    env.current_step = test_start + (episode * 100)
    
    episode_reward = 0
    episode_length = 0
    actions = []
    
    for _ in range(200):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        episode_reward += reward
        episode_length += 1
        actions.append(action)
        
        if done or truncated:
            break
    
    final_equity = env.equity
    pnl_pct = (final_equity - 10000) / 10000 * 100
    
    # Calculate average leverage and position
    avg_lev = np.mean([3.0 + a[0] * 27.0 for a in actions])
    avg_pos = np.mean([a[1] for a in actions])
    
    results.append({
        'episode': episode + 1,
        'reward': episode_reward,
        'length': episode_length,
        'final_equity': final_equity,
        'pnl_pct': pnl_pct,
        'avg_leverage': avg_lev,
        'avg_position': avg_pos
    })
    
    print(f"Episode {episode+1:2d}: Reward={episode_reward:7.1f} | Equity=${final_equity:,.2f} | PnL={pnl_pct:+6.2f}% | Lev={avg_lev:4.1f}x | Pos={avg_pos*100:4.1f}%")

# Summary
print("\n" + "=" * 60)
print(" AGGRESSIVE Agent Performance")
print("=" * 60)

avg_pnl = np.mean([r['pnl_pct'] for r in results])
avg_lev = np.mean([r['avg_leverage'] for r in results])
avg_pos = np.mean([r['avg_position'] for r in results])
win_rate = sum(1 for r in results if r['pnl_pct'] > 0) / len(results)

print(f"\n Returns:")
print(f"   Average PnL: {avg_pnl:+.2f}%")
print(f"   Win Rate: {win_rate*100:.1f}%")
print(f"   Best: {max(r['pnl_pct'] for r in results):+.2f}%")
print(f"   Worst: {min(r['pnl_pct'] for r in results):+.2f}%")

print(f"\n Strategy:")
print(f"   Avg Leverage: {avg_lev:.1f}x")
print(f"   Avg Position: {avg_pos*100:.1f}%")

print(f"\n vs Conservative:")
print(f"   Aggressive PnL: {avg_pnl:+.2f}%")
print(f"   Conservative PnL: +0.06%")
print(f"   Leverage Delta: {avg_lev:.1f}x vs 3.0x")
print(f"   Position Delta: {avg_pos*100:.1f}% vs 2.5%")

print("\n" + "=" * 60)
if avg_pnl > 0.06:
    print(" Aggressive agent outperforms conservative!")
else:
    print("  Conservative agent remains superior")
print("=" * 60)
