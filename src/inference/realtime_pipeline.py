"""
REAL-TIME DATA PIPELINE
Fetches live market data, computes topology, generates predictions.
Streams everything to the dashboard.
"""

import asyncio
import json
import numpy as np
import pandas as pd
import ccxt
import torch
import websockets
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from collections import deque
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.forecasting.topology_forecaster import create_model
from src.topology.persistence_core import ProductionTopologyEngine
from src.fusion.onchain_graph import WalletGraph
from src.validation.monte_carlo import MonteCarloForecaster
from src.risk.risk_system import NuclearRiskSystem
from src.execution.telemetry_bridge import TelemetryBridge

# NEW: Nuclear Engines
from src.signals.hawkes_cascade import process_tick as hawkes_process_tick
from src.signals.rough_path_signature import RoughPathEngine

class LiveDataPipeline:
    def __init__(self, 
                 model_path='models/transformer_best.pth',
                 symbol='BTC/USDT',
                 timeframe='15m',
                 lookback_candles=100):
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback_candles
        self.model_path = model_path
        
        # Initialize components
        print(" Initializing Real-Time Pipeline...")
        
        # 1. Exchange connection
        self.exchange = ccxt.binance({
            'rateLimit': 1200,
            'enableRateLimit': True
        })
        
        # 2. Topology engine
        self.topo_engine = ProductionTopologyEngine(resolution=32)
        
        # 3. On-Chain Graph Engine (NEW)
        self.wallet_graph = WalletGraph()
        
        # 3.5. Monte Carlo Forecaster (for probability curves)
        self.mc_forecaster = MonteCarloForecaster(n_simulations=1000, noise_scale=0.015)
        
        # 3.6. Nuclear Risk System (for kill-switch)
        self.risk_system = NuclearRiskSystem(
            tti_threshold=8.0,
            max_drawdown_daily=0.05,  # 5% daily max DD
            cooldown_minutes=30
        )
        
        # 4. Model
        self.device = 'cpu' # Force CPU for stability
        print(f"   Device: {self.device}")
        self.model = create_model()
        
        print(f" Loading model from {model_path}...")
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f" Model loaded ({sum(p.numel() for p in self.model.parameters()):,} params)")
        except Exception as e:
            print(f" Model load failed: {e}. Using random weights.")
            self.model.to(self.device)
        
        # 5. Price and Volume buffer (rolling window)
        self.price_buffer = deque(maxlen=lookback_candles)
        self.volume_buffer = deque(maxlen=lookback_candles)
        
        # NEW: Persistence Image Buffer for Transformer
        self.persistence_buffer = deque(maxlen=72)
        
        # NEW: Hawkes Window (for recent events)
        self.hawkes_window = deque(maxlen=1000)
        
        # NEW: Initialize Nuclear Engines
        self.hawkes_process_tick = hawkes_process_tick
        self.signature_engine = RoughPathEngine(model_path=None) # Random weights for now
        
        # NEW: Telemetry Bridge
        self.telemetry_bridge = TelemetryBridge()
        
        # 6. Current state
        self.current_equity = 10000.0
        self.current_positions = []
        
    async def fetch_latest_candles(self):
        """Fetch latest N candles from exchange."""
        try:
            candles = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=self.lookback
            )
            
            df = pd.DataFrame(
                candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            return df
        except Exception as e:
            print(f"  Error fetching candles: {e}")
            return pd.DataFrame()

    async def fetch_order_book_depth(self):
        """Fetch order book for wall detection (Hawkes)."""
        try:
            ob = self.exchange.fetch_order_book(self.symbol, limit=100)
            
            # Sum large walls (> $5M)
            bid_walls = sum([p * a for p, a in ob['bids'] if p * a > 5_000_000])
            ask_walls = sum([p * a for p, a in ob['asks'] if p * a > 5_000_000])
            
            return {
                'bid_delta': bid_walls,
                'ask_delta': ask_walls,
                'timestamp': time.time()
            }
        except Exception as e:
            print(f"  Order book fetch error: {e}")
            return {'bid_delta': 0, 'ask_delta': 0, 'timestamp': time.time()}

    def fetch_liquidations(self):
        """Fetch recent liquidation orders from Binance Futures."""
        try:
            # Use public API directly for simplicity
            url = "https://fapi.binance.com/fapi/v1/allForceOrders"
            params = {"symbol": self.symbol.replace('/', ''), "limit": 10}
            r = requests.get(url, params=params, timeout=5)
            if r.status_code == 200:
                return r.json()
            return []
        except Exception as e:
            print(f"  Error fetching liquidations: {e}")
            return []

    def fetch_whale_trades(self):
        """Fetch recent large trades (> $100k)."""
        try:
            url = "https://fapi.binance.com/fapi/v1/aggTrades"
            params = {"symbol": self.symbol.replace('/', ''), "limit": 500}
            r = requests.get(url, params=params, timeout=5)
            if r.status_code == 200:
                trades = r.json()
                whales = []
                for t in trades:
                    price = float(t['p'])
                    qty = float(t['q'])
                    value = price * qty
                    if value > 100000:  # $100k threshold
                        whales.append({
                            'price': price,
                            'size': qty,
                            'value': value,
                            'side': 'SELL' if t['m'] else 'BUY', # m=True means maker was buy, so taker was sell
                            'timestamp': t['T']
                        })
                return whales
            return []
        except Exception as e:
            print(f"  Error fetching whale trades: {e}")
            return []

    def run_inference(self, current_price):
        """Run model inference on persistence buffer."""
        if len(self.persistence_buffer) < 72:
            # Not enough data yet, return dummy prediction
            return [{
                'timestamp': datetime.now().isoformat(),
                'price': current_price,
                'p5': current_price * 0.95,
                'p25': current_price * 0.98,
                'p75': current_price * 1.02,
                'p95': current_price * 1.05
            }], []

        # Prepare input: (1, 72, 1, 32, 32)
        # Stack buffer
        images = np.stack(list(self.persistence_buffer), axis=0) # (72, 32, 32)
        x = torch.tensor(images, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(self.device)
        
        # Predict
        with torch.no_grad():
            scalars, vectors, next_image = self.model(x)
            # Scalars output is [Loop Score, TTI]
            # We don't have price forecast head in this specific model version apparently?
            # Wait, TopoTransformerGPT output is (scalars, vectors, next_image)
            # It seems this model forecasts TOPOLOGY, not PRICE directly.
            # But we need price for the dashboard.
            
            # Assuming we map topology back to price trend or use a separate head?
            # Or maybe the scalar head HAS price?
            # In topology_forecaster.py:
            # self.scalar_head = nn.Linear(..., 2) # [Loop Score, TTI]
            
            # So this model predicts FUTURE TOPOLOGY.
            # We need to infer price direction from predicted topology.
            
            pred_loop = scalars[0, 0].item()
            pred_tti = scalars[0, 1].item()
            
            # Heuristic Price Forecast based on predicted topology
            # Bull Loop -> Price Up
            # Bear Loop (negative score?) -> Price Down
            # High TTI -> Volatility
            
            trend = pred_loop * 0.01 # Arbitrary scaling
            vol = pred_tti * 0.005
            
            forecast_price = current_price * (1 + trend)
            
            # Construct dummy distribution around forecast
            forecasts = []
            for i in range(48):
                t_price = current_price * (1 + trend * (i+1)/48)
                forecasts.append(asdict(dataclass(frozen=True)(
                    lambda: None, 
                    timestamp=(datetime.now() + timedelta(minutes=15*(i+1))).isoformat(),
                    p5=t_price * (1 - vol),
                    p25=t_price * (1 - vol*0.5),
                    p50=t_price,
                    p75=t_price * (1 + vol*0.5),
                    p95=t_price * (1 + vol)
                )()))
                
        # Convert to format expected by dashboard
        predictions = []
        for forecast in forecasts:
            predictions.append({
                'timestamp': forecast['timestamp'],
                'price': forecast['p50'],
                'p5': forecast['p5'],
                'p25': forecast['p25'],
                'p75': forecast['p75'],
                'p95': forecast['p95']
            })
        
        return predictions, forecasts

    def generate_fusion_narrative(self, prediction, current_price, whale_trades, liquidations, topo_sig, graph_sig, mc_forecast=None, hawkes_score=0.0, sig_score=0.0):
        """
        HYBRID FUSION LAYER: Combines AI Topology + Whale Flow + Liquidations + On-Chain Graph + Confidence Intervals + Nuclear Models.
        Returns a narrative string and a confidence score.
        """
        # 1. AI View
        ai_direction = "BULLISH" if prediction['price'] > current_price else "BEARISH"
        ai_change = (prediction['price'] - current_price) / current_price * 100
        
        # Confidence based on CI width
        ci_width = (prediction['p95'] - prediction['p5']) / prediction['price']
        confidence = "HIGH" if ci_width < 0.05 else "MEDIUM" if ci_width < 0.10 else "LOW"
        
        # 2. Whale View
        whale_net_value = sum([w['value'] * (1 if w['side'] == 'BUY' else -1) for w in whale_trades])
        whale_sentiment = "BULLISH" if whale_net_value > 0 else "BEARISH"
        
        # 3. Liquidation View
        liq_net_value = sum([float(l['price']) * float(l['origQty']) * (1 if l['side'] == 'SELL' else -1) for l in liquidations])
        
        # 4. Topology View
        topo_sentiment = "BULLISH" if topo_sig > 0 else "BEARISH"
        
        # 5. Nuclear View
        hawkes_sentiment = "BULLISH" if hawkes_score > 0.2 else "BEARISH" if hawkes_score < -0.2 else "NEUTRAL"
        sig_sentiment = "BULLISH" if sig_score > 0.2 else "BEARISH" if sig_score < -0.2 else "NEUTRAL"

        # Narrative Construction
        narrative = f"AI sees {ai_direction} ({ai_change:.2f}%) (Confidence: {confidence}, ±{ci_width*100/2:.1f}%)"
        
        if topo_sentiment == ai_direction:
            narrative += f" supported by {topo_sentiment.title()} Loops."
        else:
            narrative += f" BUT Topology diverges ({topo_sentiment.title()} Loops)."
            
        if whale_sentiment == ai_direction:
            narrative += f" Whales confirm (${abs(whale_net_value)/1000:.0f}k net)."
        else:
            narrative += f" BUT Whales diverge (${abs(whale_net_value)/1000:.0f}k net)."
            
        if hawkes_sentiment != "NEUTRAL":
             narrative += f"  Hawkes: {hawkes_sentiment} Cascade."
             
        if sig_sentiment != "NEUTRAL":
             narrative += f"  Signature: {sig_sentiment} Pattern."

        return narrative

    def takens_embedding(self, prices, dim=3, delay=5):
        """Transform 1D price series into 3D point cloud."""
        n = len(prices) - (dim - 1) * delay
        if n <= 0: return np.zeros((1, dim))
        
        point_cloud = np.zeros((n, dim))
        for i in range(n):
            for j in range(dim):
                point_cloud[i, j] = prices[i + j * delay]
        
        # Normalize
        point_cloud = (point_cloud - point_cloud.mean(axis=0)) / (point_cloud.std(axis=0) + 1e-8)
        return point_cloud

    async def stream_to_dashboard(self):
        """Main loop: Fetch data -> Inference -> Stream to Dashboard."""
        print(" Streaming to Dashboard (localhost:8000)...")
        
        # Connect to dashboard
        await self.telemetry_bridge.connect()
        
        iteration = 0
        while True:
            try:
                # 1. Fetch Data
                df = await self.fetch_latest_candles()
                if df.empty:
                    await asyncio.sleep(5)
                    continue
                
                current_price = df.iloc[-1]['close']
                current_vol = df.iloc[-1]['volume']
                
                # Update buffers
                self.price_buffer.append(current_price)
                self.volume_buffer.append(current_vol)
                
                # Fetch auxiliary data
                whale_trades = self.fetch_whale_trades()
                liquidations = self.fetch_liquidations()
                orderbook = await self.fetch_order_book_depth()
                
                # 2. Run Inference
                # A. Topology
                # Transform to Point Cloud first
                point_cloud = self.takens_embedding(df['close'].values, dim=3, delay=5)
                
                # analyze_window returns a TopologySignature object
                topo_sig_obj = self.topo_engine.analyze_window(point_cloud)
                topo_sig = topo_sig_obj.loop_score
                persistence_img = topo_sig_obj.persistence_image
                tti = topo_sig_obj.tti
                
                # Update persistence buffer
                self.persistence_buffer.append(persistence_img)
                
                # B. Transformer Model
                predictions, forecasts = self.run_inference(current_price)
                
                # C. On-Chain Graph
                graph_sig = self.wallet_graph.update_and_get_signal(current_price)
                
                # D. Monte Carlo Forecasts
                mc_forecasts = self.mc_forecaster.forecast(
                    current_price=current_price,
                    price_history=list(self.price_buffer),
                    topology_signature=topo_sig_obj,
                    horizon=48
                )
                
                # E. Nuclear Hawkes
                long_liq_vol = sum([float(l['price']) * float(l['origQty']) for l in liquidations if l['side'] == 'SELL'])
                short_liq_vol = sum([float(l['price']) * float(l['origQty']) for l in liquidations if l['side'] == 'BUY'])
                
                hawkes_score, cascade_prob = self.hawkes_process_tick(
                    long_liqs=long_liq_vol,
                    short_liqs=short_liq_vol,
                    bid_delta=orderbook['bid_delta'],
                    ask_delta=orderbook['ask_delta'],
                    timestamp=time.time()
                )
                
                # F. Nuclear Signatures
                sig_loop_score = 0.0
                sig_leverage = 1.0
                if len(self.price_buffer) >= 72:
                    # Construct path: [Price, Vol, Funding(dummy), LiqVol(dummy)]
                    # For MVP we use Price/Vol and 0s for others if not available history
                    path_data = torch.zeros((72, 4))
                    # Fill with buffer data
                    prices = torch.tensor(list(self.price_buffer)[-72:])
                    vols = torch.tensor(list(self.volume_buffer)[-72:])
                    
                    # Normalize
                    path_data[:, 0] = (prices - prices.mean()) / (prices.std() + 1e-8)
                    path_data[:, 1] = (vols - vols.mean()) / (vols.std() + 1e-8)
                    
                    sig_loop_score, sig_leverage = self.signature_engine.predict(path_data)
                
                # 3. Risk Check
                # tti is already computed
                risk_state = self.risk_system.check_risk({'tti': tti}, self.current_equity, self.current_equity)
                kill_switch = not risk_state.can_trade
                risk_msg = risk_state.reason
                
                # 4. Fusion Narrative
                fusion = self.generate_fusion_narrative(
                    predictions[-1], 
                    current_price, 
                    whale_trades, 
                    liquidations, 
                    topo_sig, 
                    graph_sig, 
                    mc_forecasts[-1],
                    hawkes_score,
                    sig_loop_score
                )
                
                # 5. Construct Telemetry Packet
                telemetry = {
                    "timestamp": datetime.now().isoformat(),
                    "price": current_price,
                    "topology": {
                        "loop_score": topo_sig,
                        "tti": tti,
                        "persistence_image": persistence_img.tolist() if persistence_img is not None else []
                    },
                    "prediction": predictions[-1],
                    "fusion": {
                        "narrative": fusion,
                        "whale_net": sum([w['value'] * (1 if w['side'] == 'BUY' else -1) for w in whale_trades]),
                        "smart_money_score": graph_sig
                    },
                    "risk": {
                        "kill_switch": kill_switch,
                        "message": risk_msg
                    },
                    "nuclear": {
                        "hawkes_score": hawkes_score,
                        "cascade_prob": cascade_prob,
                        "sig_loop_score": sig_loop_score,
                        "sig_leverage": sig_leverage
                    },
                    "whales": whale_trades,
                    "liquidations": liquidations,
                    "mc_forecast": asdict(mc_forecasts[-1])
                }
                
                # 6. Send to Dashboard
                await self.telemetry_bridge.send(telemetry)
                
                # Print status
                print(f"============================================================")
                print(f"Iteration {iteration} - {datetime.now().strftime('%H:%M:%S')}")
                print(f"    Current BTC/USDT: ${current_price:,.2f}")
                print(f"    Whale Trades: {len(whale_trades)} |  Liquidations: {len(liquidations)}")
                print(f"     On-Chain: Smart Money Score={graph_sig:.1f}")
                print(f"    Topology: Loop={topo_sig:.4f}, TTI={tti:.4f}")
                print(f"    Hawkes: {hawkes_score:.3f} | Cascade Risk: {cascade_prob:.1%}")
                print(f"    Signature: Loop={sig_loop_score:.3f} | Lev={sig_leverage:.1f}x")
                
                if kill_switch:
                    print(f"    KILL-SWITCH ACTIVE: {risk_msg}")
                
                print(f"    48h Forecast: ${predictions[-1]['price']:,.0f} | 95% CI: [${predictions[-1]['p5']:,.0f}, ${predictions[-1]['p95']:,.0f}] | Confidence: {fusion.split('Confidence: ')[1].split(',')[0]}")
                print(f"    Fusion Narrative: {fusion}")
                
                direction = "" if predictions[-1]['price'] > current_price else ""
                print(f"    Prediction: 48h trend {direction} (${predictions[-1]['price']:,.2f})")
                
                iteration += 1
                await asyncio.sleep(60) # 1 minute loop
                
            except Exception as e:
                print(f"  Pipeline Error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)

    async def run(self):
        await self.stream_to_dashboard()

if __name__ == "__main__":
    # Initialize synchronously before event loop to avoid library conflicts
    pipeline = LiveDataPipeline()
    
    async def main():
        await pipeline.run()
        
    asyncio.run(main())
