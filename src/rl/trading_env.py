import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TradingEnv(gym.Env):
    """
    Custom Gymnasium Environment for Trading.
    
    State: [Hurst, RSI, EMA_fast, EMA_slow, ATR, Narrative_ID, Has_Zigzag_Low, Has_Zigzag_High]
    Action: 0=NEUTRAL, 1=LONG, 2=SHORT
    Reward: Change in PnL (simplified for MVP)
    """
    
    def __init__(self, df=None):
        super().__init__()
        
        self.df = df
        self.current_step = 0
        
        # State space: 8 continuous features
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, -10.0, 0.0, 0.0]),
            high=np.array([1.0, 100.0, 1000.0, 1000.0, 100.0, 10.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Action space: 3 discrete actions
        self.action_space = spaces.Discrete(3)
        
        self.current_state = None
        self.current_position = 0  # 0=NEUTRAL, 1=LONG, -1=SHORT
        self.entry_price = 0.0
        self.current_price = 100.0
        self.pnl = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_position = 0
        self.entry_price = 0.0
        self.pnl = 0.0
        
        if self.df is not None and len(self.df) > 100:
            # Pick random start point, leaving room for history
            self.current_step = np.random.randint(0, len(self.df) - 100)
            self._update_state_from_df()
        else:
            self.current_step = 0
            self.current_state = self.observation_space.sample()
            self.current_price = 100.0
        
        return self.current_state, {}
    
    def step(self, action):
        # action: 0=NEUTRAL, 1=LONG, 2=SHORT
        
        reward = 0.0
        
        if self.df is not None:
            # Historical Replay Mode
            if self.current_step >= len(self.df) - 1:
                terminated = True
                return self.current_state, 0, terminated, False, {}
            
            # Get price change from NEXT candle
            current_price = self.df.iloc[self.current_step]['c']
            next_price = self.df.iloc[self.current_step + 1]['c']
            price_change = next_price - current_price
            self.current_price = next_price
            
            # Calculate Reward
            if self.current_position == 1:  # LONG
                reward = price_change
            elif self.current_position == -1:  # SHORT
                reward = -price_change
            
            # Transaction Costs (Simplified)
            if action != 0 and ( (action == 1 and self.current_position != 1) or (action == 2 and self.current_position != -1) ):
                reward -= 0.0005 * current_price # 0.05% fee
                
            # Update Position
            if action == 0:
                self.current_position = 0
            elif action == 1:
                self.current_position = 1
            elif action == 2:
                self.current_position = -1
                
            self.pnl += reward
            
            # Move to next step
            self.current_step += 1
            self._update_state_from_df()
            
            terminated = False
            truncated = False
            
        else:
            # Random/Dummy Mode (Original)
            price_change = np.random.normal(0, 1)
            self.current_price += price_change
            
            if self.current_position == 1: reward = price_change
            elif self.current_position == -1: reward = -price_change
            
            if action == 0: self.current_position = 0
            elif action == 1: self.current_position = 1
            elif action == 2: self.current_position = -1
            
            self.pnl += reward
            self.current_state = self.observation_space.sample()
            terminated = False
            truncated = False
        
        return self.current_state, reward, terminated, truncated, {}
    
    def _update_state_from_df(self):
        # Extract state from current row of df
        # Assumes df has columns: ['hurst', 'rsi', 'ema_fast', 'ema_slow', 'atr', 'narrative', 'zigzag_low', 'zigzag_high']
        row = self.df.iloc[self.current_step]
        self.current_state = np.array([
            row['hurst'],
            row['rsi'],
            row['ema_fast'],
            row['ema_slow'],
            row['atr'],
            row['narrative'],
            row['zigzag_low'],
            row['zigzag_high']
        ], dtype=np.float32)

    
    def set_state(self, state):
        """Set state from real market data"""
        self.current_state = np.array(state, dtype=np.float32)
        
    def render(self):
        pass
