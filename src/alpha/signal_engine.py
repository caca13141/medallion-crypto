from src.alpha.factors.momentum import calculate_ema, calculate_rsi
from src.alpha.factors.volatility import calculate_atr
from src.alpha.factors.structure import calculate_zigzag
from src.alpha.factors.persistence import calculate_hurst
from src.alpha.factors.persistence import calculate_hurst
from src.alpha.models.transformer import TopologyTransformer
from src.alpha.models.loss import OrdinalWassersteinLoss
from src.alpha.factors.narrative import NarrativeMapper
from src.alpha.factors.fusion import FusionFactor
from src.config import Config
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os

class SignalEngine:
    def __init__(self):
        self.device = torch.device("cpu") # Force CPU for safety
        self.model = TopologyTransformer(
            input_dim=Config.TRANSFORMER_INPUT_DIM,
            d_model=Config.TRANSFORMER_D_MODEL,
            nhead=Config.TRANSFORMER_NHEAD,
            num_layers=Config.TRANSFORMER_LAYERS
        ).to(self.device)
        
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_wass = OrdinalWassersteinLoss(num_classes=3)
        
        self.criterion_wass = OrdinalWassersteinLoss(num_classes=3)
        
        self.narrative_mapper = NarrativeMapper()
        self.fusion_factor = FusionFactor()
        
        # PPO Agent (Optional)
        self.ppo_agent = None
        if Config.PPO_ENABLE:
            from src.rl.ppo_agent import TopologicalPPOAgent
            self.ppo_agent = TopologicalPPOAgent()
        
        self.load_model()
        self.model.eval() 

    def load_model(self):
        if os.path.exists(Config.MODEL_PATH):
            try:
                self.model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=self.device))
                print(f"BRAIN LOADED: {Config.MODEL_PATH}")
            except Exception as e:
                print(f"BRAIN LOAD FAILED: {e}")

    def save_model(self):
        try:
            os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
            torch.save(self.model.state_dict(), Config.MODEL_PATH)
        except Exception as e:
            print(f"BRAIN SAVE FAILED: {e}")

    def train_step(self, df, target_signal):
        # Online Learning Step
        # target_signal: 0=Neutral, 1=Long, 2=Short
        if len(df) < 50: return
        
        self.model.train()
        
        features = df[['c', 'o', 'h', 'l', 'v']].values
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        src = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        target = torch.tensor([target_signal], dtype=torch.long).to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(src)
        
        # Combined Loss
        loss_ce = self.criterion_ce(output, target)
        loss_wass = self.criterion_wass(output, target)
        
        loss = loss_ce + (Config.WASSERSTEIN_GAIN * loss_wass)
        
        loss.backward()
        self.optimizer.step()
        
        self.model.eval()
        return loss.item()


    def analyze(self, df, feed=None, coin=None):
        """
        Analyze the market data and generate a signal.
        """
        if df.empty or len(df) < Config.LOOKBACK:
            return 0, 0, 0, 0, 0 # Added Fusion Score return
        
        # Calculate Factors
        df['ema_fast'] = calculate_ema(df['c'], Config.EMA_FAST)
        df['ema_slow'] = calculate_ema(df['c'], Config.EMA_SLOW)
        df['rsi'] = calculate_rsi(df['c'], Config.RSI_PERIOD)
        df['atr'] = calculate_atr(df, Config.ATR_PERIOD)
        df['zigzag'] = calculate_zigzag(df, Config.ZIGZAG_THRESHOLD)
        
        last = df.iloc[-1]
        
        # Logic
        recent_pivots = df['zigzag'].iloc[-10:]
        has_recent_low = (recent_pivots == -1).any()
        has_recent_high = (recent_pivots == 1).any()
        
        # Transformer Inference
        # Prepare input tensor: [1, seq_len, 5]
        features = df[['c', 'o', 'h', 'l', 'v']].values
        # Normalize (Simple Z-score for MVP)
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        src = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            prediction = self.model(src)
            # prediction shape: [1, 3] (Long, Neutral, Short)
            probs = torch.softmax(prediction, dim=1)
            model_signal = torch.argmax(probs).item() # 0: Long, 1: Neutral, 2: Short (Mapping needs definition)
            
        # Mapping: 0=LONG, 1=NEUTRAL, 2=SHORT (Arbitrary for MVP, usually 0=Neutral)
        # Let's define: 0=Neutral, 1=Long, 2=Short
        
        # Narrative ID (TDA)
        narrative_id = self.narrative_mapper.map_narrative(df)
        
        # Persistence (Hurst)
        hurst = calculate_hurst(df['c'].values)
        
        # Fusion (On-chain)
        fusion_score = 0
        funding_rate = 0.0
        if Config.FUSION_ENABLE and feed is not None and coin is not None:
            funding_df = feed.get_funding_history(coin, limit=10)
            # Get current OI (simplified, just passing 0 for now as we need to find coin index)
            # In a real run, we'd cache asset_ctxs in the engine or feed.
            # For MVP, let's fetch it if possible, or skip OI raw value and rely on funding.
            # asset_ctxs = feed.get_asset_ctxs() 
            # We'll skip raw OI for now to avoid extra API call latency in this loop, 
            # relying on Funding Rate which is the primary proxy for positioning.
            fusion_score, funding_rate = self.fusion_factor.calculate_fusion_score(df, funding_df, 0, 0)
        
        signal = 0
        # Ensemble: Zigzag + RSI + Transformer + Narrative Filter + Hurst Filter + Fusion Filter
        # If Narrative ID is -1 (Noise), we might want to reduce size or block.
        # For MVP: We just log it and proceed.
        
        # Hurst Filter: Only take trend trades if H > Threshold
        is_trending = hurst > Config.HURST_THRESHOLD
        
        # Fusion Filter: 
        # If Fusion Score is very negative (Crowded Longs), block Longs.
        # If Fusion Score is very positive (Crowded Shorts), block Shorts (or encourage Longs).
        
        # PPO Agent Decision (if enabled)
        if self.ppo_agent is not None:
            state = self.get_state(df, hurst, narrative_id, last, has_recent_low, has_recent_high, model_signal)
            ppo_action = self.ppo_agent.predict(state)
            # ppo_action: 0=NEUTRAL, 1=LONG, 2=SHORT
            signal = 1 if ppo_action == 1 else (-1 if ppo_action == 2 else 0)
        else:
            # Original rule-based logic
            if last['ema_fast'] > last['ema_slow'] and last['rsi'] > 50 and has_recent_low:
                if model_signal == 1 and is_trending: # Transformer confirms Long AND Trending
                    if fusion_score >= 0: # Don't buy if crowded longs (score < 0)
                        signal = 1
            elif last['ema_fast'] < last['ema_slow'] and last['rsi'] < 50 and has_recent_high:
                if model_signal == 2 and is_trending: # Transformer confirms Short AND Trending
                    if fusion_score <= 0: # Don't short if crowded shorts (score > 0)
                        signal = -1
            
        return signal, last['atr'], narrative_id, hurst, fusion_score

    def get_state(self, df, hurst, narrative_id, last, has_recent_low, has_recent_high, model_signal):
        """Extract state vector for PPO agent"""
        return np.array([
            hurst,
            last['rsi'],
            last['ema_fast'],
            last['ema_slow'],
            last['atr'],
            float(narrative_id),
            1.0 if has_recent_low else 0.0,
            1.0 if has_recent_high else 0.0
        ], dtype=np.float32)

