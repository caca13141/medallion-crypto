from stable_baselines3 import PPO
from src.rl.trading_env import TradingEnv
from src.config import Config
import os

class TopologicalPPOAgent:
    """
    PPO Agent for trading using topological features.
    """
    
    def __init__(self, model_path="src/data/ppo_model.zip"):
        self.env = TradingEnv()
        self.model_path = model_path
        
        # Try to load existing model, otherwise create new
        if os.path.exists(self.model_path):
            self.model = PPO.load(self.model_path, env=self.env)
            print(f"PPO MODEL LOADED: {self.model_path}")
        else:
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=Config.PPO_LEARNING_RATE,
                gamma=Config.PPO_GAMMA,
                gae_lambda=Config.PPO_GAE_LAMBDA,
                clip_range=Config.PPO_CLIP_RANGE,
                verbose=0
            )
            print("PPO MODEL INITIALIZED (NEW)")
    
    def predict(self, state):
        """
        Predict action given state.
        state: [Hurst, RSI, EMA_fast, EMA_slow, ATR, Narrative_ID, Has_Zigzag_Low, Has_Zigzag_High]
        returns: action (0=NEUTRAL, 1=LONG, 2=SHORT)
        """
        action, _ = self.model.predict(state, deterministic=True)
        return int(action)
    
    def train_step(self, total_timesteps=1):
        """
        Train the model for a few timesteps.
        For online learning, we can call this periodically.
        """
        self.model.learn(total_timesteps=total_timesteps)
    
    def save(self):
        """Save the model"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        print(f"PPO MODEL SAVED: {self.model_path}")
