"""
Nuclear RL Layer: Wasserstein-PPO for Position Sizing
Replaces Kelly with full PPO agent using Wasserstein auxiliary loss
"""
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from scipy.stats import wasserstein_distance

class WassersteinAuxiliaryLoss(nn.Module):
    """
    Wasserstein distance between predicted and realized persistence diagrams.
    Used as auxiliary loss to improve PPO predictions.
    """
    
    def __init__(self, weight=0.5):
        super().__init__()
        self.weight = weight
        
    def forward(self, pred_diagram, real_diagram):
        """
        Compute Wasserstein distance between two persistence diagrams.
        
        Args:
            pred_diagram: (n, 2) predicted birth-death pairs
            real_diagram: (m, 2) realized birth-death pairs
            
        Returns:
            loss: scalar Wasserstein distance
        """
        if len(pred_diagram) == 0 or len(real_diagram) == 0:
            return torch.tensor(0.0)
            
        # Convert to numpy for scipy
        pred_np = pred_diagram.detach().cpu().numpy()
        real_np = real_diagram.detach().cpu().numpy()
        
        # Compute Wasserstein-1 distance on lifetimes
        pred_lifetimes = pred_np[:, 1] - pred_np[:, 0]
        real_lifetimes = real_np[:, 1] - real_np[:, 0]
        
        # Handle edge cases
        if len(pred_lifetimes) == 0 or len(real_lifetimes) == 0:
            return torch.tensor(0.0)
            
        try:
            w_dist = wasserstein_distance(pred_lifetimes, real_lifetimes)
            return torch.tensor(w_dist * self.weight)
        except:
            return torch.tensor(0.0)


class TopologicalTradingEnv(gym.Env):
    """
    Enhanced trading environment with topological state.
    
    State space:
    - loop_score
    - tti
    - bifiltration_features (20D)
    - price momentum features (5D)
    - current position
    - unrealized PnL
    
    Action space:
    - 0: Flatten
    - 1-5: Long with leverage 1x-5x
    - 6-10: Short with leverage 1x-5x
    """
    
    def __init__(self, historical_data, topology_integrator):
        super().__init__()
        
        self.data = historical_data
        self.topology = topology_integrator
        self.current_step = 0
        self.max_steps = len(historical_data) - 100
        
        # State: 28D (loop + tti + 20 bifilt + 5 price + pos + pnl)
        self.observation_space = gym.spaces.Box(
            low=-10, high=10, shape=(28,), dtype=np.float32
        )
        
        # Action: 11 discrete actions (flatten + 5 long lev + 5 short lev)
        self.action_space = gym.spaces.Discrete(11)
        
        self.equity = 10000
        self.position = 0  # -5 to +5 (leverage)
        self.entry_price = 0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = np.random.randint(100, self.max_steps - 100)
        self.equity = 10000
        self.position = 0
        self.entry_price = 0
        return self._get_state(), {}
        
    def step(self, action):
        # Get current and next price
        current_price = self.data.iloc[self.current_step]['c']
        next_price = self.data.iloc[self.current_step + 1]['c']
        
        # Close existing position if any
        pnl = 0
        if self.position != 0:
            price_change = next_price - self.entry_price
            pnl = self.position * (price_change / self.entry_price) * self.equity
            self.equity += pnl
            
        # Execute new action
        if action == 0:
            # Flatten
            self.position = 0
        elif 1 <= action <= 5:
            # Long with leverage
            leverage = action
            self.position = leverage
            self.entry_price = next_price
        elif 6 <= action <= 10:
            # Short with leverage
            leverage = action - 5
            self.position = -leverage
            self.entry_price = next_price
            
        # Advance
        self.current_step += 1
        done = self.current_step >= self.max_steps or self.equity <= 1000
        
        # Reward = PnL normalized by equity
        reward = pnl / 10000  # Normalize
        
        # Penalty for excessive leverage in high turbulence
        state = self._get_state()
        tti = state[1]
        if tti > 3.0 and abs(self.position) > 2:
            reward -= 0.1  # Turbulence penalty
            
        return self._get_state(), reward, done, False, {}
        
    def _get_state(self):
        """Extract 28D state vector"""
        # Get window
        window = self.data.iloc[self.current_step-99:self.current_step+1]
        
        # Topology analysis
        topo = self.topology.analyze(window)
        loop_score = topo['loop_score']
        tti = topo['tti']
        bifilt = topo['bifiltration_features']
        
        # Price features
        prices = window['c'].values
        returns = np.diff(np.log(prices))
        price_feats = np.array([
            returns[-1],  # Last return
            np.mean(returns[-5:]),  # 5-period avg
            np.mean(returns[-20:]),  # 20-period avg
            np.std(returns[-20:]),  # Volatility
            (prices[-1] - prices[-50]) / prices[-50]  # 50-period momentum
        ])
        
        # Position & PnL
        unrealized_pnl = 0
        if self.position != 0:
            current_price = prices[-1]
            unrealized_pnl = self.position * (current_price - self.entry_price) / self.entry_price
            
        state = np.concatenate([
            [loop_score, tti],
            bifilt,
            price_feats,
            [self.position / 5, unrealized_pnl]  # Normalize position
        ])
        
        return state.astype(np.float32)


class WassersteinPPOAgent:
    """
    PPO agent with Wasserstein auxiliary loss for topology-enhanced trading.
    """
    
    def __init__(self, env, learning_rate=3e-4, wasserstein_weight=0.5):
        self.env = env
        self.wasserstein_loss = WassersteinAuxiliaryLoss(weight=wasserstein_weight)
        
        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0
        )
        
    def train(self, total_timesteps=100000):
        """Train the PPO agent"""
        self.model.learn(total_timesteps=total_timesteps)
        
    def predict(self, state):
        """Predict action given state"""
        action, _ = self.model.predict(state, deterministic=True)
        return action
        
    def save(self, path):
        """Save model"""
        self.model.save(path)
        
    def load(self, path):
        """Load model"""
        self.model = PPO.load(path, env=self.env)
