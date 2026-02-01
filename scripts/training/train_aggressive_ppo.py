"""
Train Aggressive PPO Agent
Modified reward function for higher-risk, higher-return strategy.
"""
import sys
sys.path.append('/Users/raphaelmaksoud/crypto toppo')

from src.rl.continuous_ppo import ContinuousTopoEnv, TopoFeatureExtractor, WassersteinRewardCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym
import numpy as np

class AggressiveTopoEnv(ContinuousTopoEnv):
    """
    Aggressive trading environment with modified reward structure.
    Encourages higher leverage and larger positions.
    """
    
    def step(self, action):
        """Execute step with aggressive reward."""
        # Parse action
        leverage_factor = action[0]  # 0-1
        position_fraction = action[1]  # 0-1
        
        leverage = 3.0 + leverage_factor * 27.0  # Map to 3x-30x
        position_size = position_fraction
        
        # Get current state
        current = self.data[self.current_step + self.lookback]
        next_step = self.data[self.current_step + self.lookback + 1]
        
        current_price = current['price']
        next_price = next_step['price']
        
        # Compute return
        price_return = (next_price - current_price) / current_price
        
        # Apply leverage and position size
        pnl = self.equity * position_size * leverage * price_return
        
        # Update equity
        self.equity += pnl
       
        # Drawdown
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = (self.peak_equity - self.equity) / self.peak_equity
        
        # AGGRESSIVE REWARD STRUCTURE
        # 1. Amplify profit rewards (2x multiplier)
        profit_reward = (pnl / self.equity) * 100.0  # Scale up profits
        
        # 2. Reduce drawdown penalty (50% of conservative)
        drawdown_penalty = -drawdown * 5.0 if drawdown > 0.10 else 0.0  # Only penalize >10% DD
        
        # 3. Bonus for using higher leverage
        leverage_bonus = (leverage - 3.0) / 27.0 * 0.5  # Reward for using available leverage
        
        # 4. Bonus for larger positions
        position_bonus = position_size * 0.2  # Encourage full capital deployment
        
        # 5. Minimal Wasserstein penalty
        wasserstein_dist = current['wasserstein_dist']
        wasserstein_penalty = -wasserstein_dist * 0.01  # 10x less than conservative
        
        # Combined aggressive reward
        reward = profit_reward + drawdown_penalty + leverage_bonus + position_bonus + wasserstein_penalty
        
        # Move to next step
        self.current_step += 1
        
        # Check termination (more lenient - allow bigger drawdowns)
        done = (self.current_step >= self.max_steps - 1) or (self.equity < 3000)  # 70% loss allowed
        truncated = False
        
        # Get next observation
        obs = self._get_obs()
        
        info = {
            'wasserstein_dist': wasserstein_dist,
            'equity': self.equity,
            'drawdown': drawdown,
            'pnl': pnl,
            'leverage': leverage,
            'position_size': position_size
        }
        
        return obs, reward, done, truncated, info

def make_aggressive_env():
    return AggressiveTopoEnv()

if __name__ == "__main__":
    print("=" * 60)
    print(" Training AGGRESSIVE PPO Agent")
    print("=" * 60)
    print("\nModifications from conservative version:")
    print("   2x profit rewards")
    print("   50% lower drawdown penalties")
    print("   Bonus for higher leverage")
    print("   Bonus for larger positions")
    print("   70% drawdown tolerance (vs 50%)")
    print("\n" + "=" * 60 + "\n")
    
    # Create vectorized environments
    print(" Initializing 4 parallel environments...")
    env = SubprocVecEnv([make_aggressive_env for _ in range(4)])
    
    # PPO with aggressive hyperparameters
    policy_kwargs = dict(
        features_extractor_class=TopoFeatureExtractor,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=5e-4,  # Higher LR for faster adaptation
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02  # Higher entropy for more exploration
    )
    
    print("\n Agent initialized")
    print("\n Starting training (150K timesteps for more exploration)...")
    print("=" * 60 + "\n")
    
    # Train with checkpoints
    total_timesteps = 150000
    save_every = 37500  # 4 checkpoints
    
    for checkpoint in range(0, total_timesteps, save_every):
        steps_remaining = min(save_every, total_timesteps - checkpoint)
        
        print(f"\n Checkpoint {checkpoint//save_every + 1}/4")
        print(f"   Training for {steps_remaining:,} steps...")
        
        callback = WassersteinRewardCallback()
        model.learn(total_timesteps=steps_remaining, reset_num_timesteps=False, callback=callback)
        
        # Save checkpoint
        checkpoint_path = f"src/data/ppo_aggressive_checkpoint_{checkpoint + steps_remaining}.zip"
        model.save(checkpoint_path)
        print(f"    Saved: {checkpoint_path}")
    
    # Final save
    final_path = "src/data/ppo_aggressive.zip"
    model.save(final_path)
    print(f"\n Aggressive training complete! Model: {final_path}")
    print("=" * 60)
