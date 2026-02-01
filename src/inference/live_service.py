"""
LIVE INFERENCE SERVICE
Loads the trained Transformer model and generates real-time predictions.
Streams predictions to the dashboard via WebSocket.
"""

import torch
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.forecasting.topology_forecaster import create_model
from src.execution.telemetry_bridge import TelemetryBridge

class LiveInferenceService:
    def __init__(self, model_path='models/transformer_best.pth'):
        self.model_path = model_path
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f" Initializing Inference Service on {self.device}...")
        
    def load_model(self):
        """Load the trained Transformer model."""
        print(f" Loading model from {self.model_path}...")
        
        self.model = create_model()
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(" Model loaded successfully!")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def generate_predictions(self, lookback_hours=24, forecast_hours=48):
        """
        Generate predictions for the next 48 hours.
        
        For now, uses mock topology data. In production, this would:
        1. Fetch recent price data
        2. Compute persistence diagrams
        3. Feed to Transformer
        4. Return predictions
        """
        
        # Mock data for demonstration
        # In production, replace with actual topology computation
        base_price = 43500 + np.random.randn() * 100
        
        predictions = []
        actuals = []
        
        now = datetime.now()
        
        # Generate 48h predictions
        for i in range(-lookback_hours, forecast_hours):
            timestamp = now + timedelta(hours=i)
            
            # Mock prediction (in production, use model.forward())
            # Add trend + noise
            predicted_price = base_price + i * 5 + np.sin(i / 8) * 200 + np.random.randn() * 50
            
            # Actual price (only for past data)
            if i <= 0:
                actual_price = base_price + i * 5 + np.sin(i / 8) * 200 + np.random.randn() * 30
                actuals.append({
                    'timestamp': timestamp.timestamp(),
                    'price': actual_price
                })
            
            predictions.append({
                'timestamp': timestamp.timestamp(),
                'price': predicted_price
            })
        
        return predictions, actuals
    
    async def stream_to_dashboard(self):
        """Stream predictions to the dashboard."""
        bridge = TelemetryBridge()
        await bridge.connect()
        
        print(" Streaming predictions to dashboard...")
        print("   Press Ctrl+C to stop.")
        
        equity = 10000.0
        
        try:
            while True:
                # Generate new predictions every 5 seconds
                predictions, actuals = self.generate_predictions()
                
                # Simulate equity changes
                equity += (np.random.random() - 0.48) * 20
                
                # Compute topology metrics (mock for now)
                persistence_data = self._mock_topology()
                
                # Build telemetry payload
                telemetry = {
                    "timestamp": datetime.now().timestamp(),
                    "pnl": round((equity - 10000) / 10000 * 100, 2),
                    "equity": round(equity, 2),
                    "drawdown": round(np.random.random() * 2, 2),
                    "tti": 1.5 + np.random.random() * 1.5,
                    "positions": self._mock_positions(),
                    "topology": persistence_data,
                    "predictions": predictions,  # NEW: 48h predictions
                    "actuals": actuals  # NEW: Historical actual prices
                }
                
                await bridge.send(telemetry)
                print(f" Sent prediction: Equity=${equity:.2f}, Next 48h trend: {'' if predictions[-1]['price'] > predictions[0]['price'] else ''}")
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
        except KeyboardInterrupt:
            print("\n⏹  Stopping inference service...")
            await bridge.close()
    
    def _mock_topology(self):
        """Generate mock topology data."""
        b0 = [np.random.random() for _ in range(int(100 + np.random.random() * 50))]
        b1 = [np.random.random() for _ in range(int(20 + np.random.random() * 10))]
        
        return {
            "persistence_image": [],
            "betti_curves": [b0, b1],
            "wasserstein_dist": 0.04 + np.random.random() * 0.02
        }
    
    def _mock_positions(self):
        """Generate mock positions."""
        return [
            {
                "symbol": "BTC-USDT",
                "side": "LONG" if np.random.random() > 0.5 else "SHORT",
                "size": round(np.random.random() * 2, 3),
                "pnl": round((np.random.random() - 0.5) * 500, 2),
                "leverage": 5
            },
            {
                "symbol": "ETH-USDT",
                "side": "LONG",
                "size": round(np.random.random() * 10, 3),
                "pnl": round((np.random.random() - 0.5) * 200, 2),
                "leverage": 3
            }
        ]


async def main():
    service = LiveInferenceService()
    service.load_model()
    await service.stream_to_dashboard()


if __name__ == "__main__":
    asyncio.run(main())
