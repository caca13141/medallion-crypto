"""
Topology-Powered Signal Engine
Replaces indicator-based system with persistent homology analysis
"""
import numpy as np
import torch
from src.topology.integrator import TopologyIntegrator
from src.forecasting.topo_transformer import TopologicalTransformer
from src.rl.wasserstein_ppo import WassersteinPPOAgent, TopologicalTradingEnv
from src.risk.nuclear_controls import NuclearRiskControls
from stable_baselines3.common.vec_env import DummyVecEnv

class TopoSignalEngine:
    """
    Nuclear signal engine powered by persistent homology.
    
    Pipeline:
    1. TopologyIntegrator → Loop Score, TTI, persistence images
    2. TopologicalTransformer → 48h H1 dissolution forecast
    3. WassersteinPPOAgent → Dynamic position sizing (1x-25x)
    4. NuclearRiskControls → TTI kill-switch, confidence caps
    """
    
    def __init__(self, enable_transformer=True, enable_ppo=True):
        # Core topology engine
        self.topology = TopologyIntegrator(lookback=100, resolution=20)
        
        # Transformer forecaster
        self.enable_transformer = enable_transformer
        self.transformer = None
        self.transformer_model = None
        if enable_transformer:
            self.transformer_model = TopologicalTransformer(
                img_size=20,
                seq_len=72,
                d_model=256,
                nhead=8,
                num_layers=6
            )
            # Load pretrained if exists
            try:
                self.transformer_model.load_state_dict(
                    torch.load('src/data/topo_transformer.pth')
                )
                self.transformer_model.eval()
            except:
                pass  # Will need training
        
        # PPO agent for position sizing
        self.enable_ppo = enable_ppo
        self.ppo_agent = None
        
        # Risk controls
        self.risk_controls = NuclearRiskControls(
            tti_threshold=3.0,
            confidence_min=0.6,
            daily_dd_limit=0.035,
            min_leverage=1,
            max_leverage=25
        )
        
        # State tracking
        self.persistence_image_buffer = []  # For transformer (72h history)
        
    def analyze(self, df, feed=None, coin=None):
        """
        Main analysis function.
        
        Args:
            df: DataFrame with OHLCV data
            feed: HyperliquidFeed instance (for on-chain data)
            coin: coin symbol
            
        Returns:
            signal: -1 (short), 0 (neutral), 1 (long)
            leverage: recommended leverage [1, 25]
            confidence: prediction confidence [0, 1]
            metadata: dict with topology metrics
        """
        if len(df) < 100:
            return 0, 1, 0.0, {}
        
        # 1. Topology analysis
        topo_result = self.topology.analyze(df)
        
        loop_score = topo_result['loop_score']
        tti = topo_result['tti']
        h0_image = topo_result['persistence_image_h0']
        h1_image = topo_result['persistence_image_h1']
        bifilt_features = topo_result['bifiltration_features']
        
        # 2. Risk control check
        # Note: daily equity tracking would come from engine
        should_flatten, reason = self.risk_controls.should_flatten(
            tti, daily_equity=10000, start_equity=10000
        )
        
        if should_flatten:
            return 0, 1, 0.0, {
                'loop_score': loop_score,
                'tti': tti,
                'regime': 'FLATTEN',
                'flatten_reason': reason
            }
        
        # 3. Transformer prediction (if enabled and buffer full)
        dissolution_time = None
        dissolution_strength = 0.0
        transformer_confidence = 0.5
        
        if self.enable_transformer and self.transformer_model is not None:
            # Add current H1 image to buffer
            self.persistence_image_buffer.append(h1_image)
            if len(self.persistence_image_buffer) > 72:
                self.persistence_image_buffer.pop(0)
            
            # Predict if buffer full
            if len(self.persistence_image_buffer) == 72:
                try:
                    # Stack to (1, 72, 1, 20, 20)
                    img_seq = np.stack(self.persistence_image_buffer)
                    img_seq = torch.from_numpy(img_seq).float()
                    img_seq = img_seq.unsqueeze(0).unsqueeze(2)  # Add batch and channel
                    
                    time_hours, strength, conf = self.transformer_model.predict(img_seq)
                    dissolution_time = float(time_hours[0])
                    dissolution_strength = float(strength[0][0])
                    transformer_confidence = float(conf[0][0])
                except Exception as e:
                    pass  # Prediction failed, use defaults
        
        # 4. Generate base signal from topology
        regime = self.topology.get_regime(loop_score, tti)
        
        signal = 0
        base_leverage = 3  # Default
        
        if regime == 'trending':
            # Low loops, low turbulence → momentum trade
            # Direction from recent price action
            recent_returns = np.diff(np.log(df.tail(20)['c'].values))
            momentum = np.sum(recent_returns)
            
            if momentum > 0.01:
                signal = 1  # Long
                base_leverage = 5
            elif momentum < -0.01:
                signal = -1  # Short
                base_leverage = 5
                
        elif regime == 'mean_reverting':
            # High loops, low turbulence → fade extremes
            recent_returns = np.diff(np.log(df.tail(20)['c'].values))
            momentum = np.sum(recent_returns)
            
            if momentum > 0.03:
                signal = -1  # Fade rally
                base_leverage = 3
            elif momentum < -0.03:
                signal = 1  # Fade selloff
                base_leverage = 3
                
        # 5. Transformer override
        # If dissolution predicted soon and strong → fade the move
        if dissolution_time is not None and dissolution_time < 12:  # <12h
            if dissolution_strength > 0.7:
                # Strong dissolution coming → prepare to fade
                if signal == 1:
                    signal = 0  # Don't buy before loop dissolves
                elif signal == -1:
                    signal = 0  # Don't short before loop dissolves
        
        # 6. PPO override (if enabled)
        if self.enable_ppo and self.ppo_agent is not None:
            try:
                # Create state vector for PPO
                state = self._create_ppo_state(
                    topo_result, df, signal, base_leverage
                )
                ppo_action = self.ppo_agent.predict(state)
                
                # Decode action
                if ppo_action == 0:
                    signal = 0
                    base_leverage = 1
                elif 1 <= ppo_action <= 5:
                    signal = 1
                    base_leverage = ppo_action
                elif 6 <= ppo_action <= 10:
                    signal = -1
                    base_leverage = ppo_action - 5
            except Exception as e:
                pass  # PPO failed, use topology signal
        
        # 7. Apply confidence-based leverage cap
        final_leverage = self.risk_controls.calculate_leverage_cap(
            transformer_confidence, base_leverage
        )
        
        metadata = {
            'loop_score': loop_score,
            'tti': tti,
            'regime': regime,
            'dissolution_time': dissolution_time,
            'dissolution_strength': dissolution_strength,
            'transformer_confidence': transformer_confidence,
            'base_leverage': base_leverage,
            'final_leverage': final_leverage,
            'bifiltration_features': bifilt_features
        }
        
        return signal, final_leverage, transformer_confidence, metadata
    
    def _create_ppo_state(self, topo_result, df, signal, leverage):
        """Create 28D state vector for PPO"""
        loop_score = topo_result['loop_score']
        tti = topo_result['tti']
        bifilt = topo_result['bifiltration_features']
        
        # Price features
        prices = df['c'].values
        returns = np.diff(np.log(prices))
        price_feats = np.array([
            returns[-1],
            np.mean(returns[-5:]),
            np.mean(returns[-20:]),
            np.std(returns[-20:]),
            (prices[-1] - prices[-50]) / prices[-50]
        ])
        
        state = np.concatenate([
            [loop_score, tti],
            bifilt,
            price_feats,
            [signal, leverage / 25]
        ])
        
        return state.astype(np.float32)
    
    def initialize_ppo(self, historical_data):
        """Initialize and train PPO agent on historical data"""
        if not self.enable_ppo:
            return
            
        # Create environment
        env = TopologicalTradingEnv(historical_data, self.topology)
        env = DummyVecEnv([lambda: env])
        
        # Create agent
        self.ppo_agent = WassersteinPPOAgent(env, learning_rate=3e-4)
        
        # Train
        print("Training PPO agent...")
        self.ppo_agent.train(total_timesteps=100000)
        
        # Save
        self.ppo_agent.save('src/data/wasserstein_ppo.zip')
        print("PPO training complete.")
    
    def save_models(self):
        """Save Transformer and PPO models"""
        if self.transformer_model is not None:
            torch.save(
                self.transformer_model.state_dict(),
                'src/data/topo_transformer.pth'
            )
        
        if self.ppo_agent is not None:
            self.ppo_agent.save('src/data/wasserstein_ppo.zip')
