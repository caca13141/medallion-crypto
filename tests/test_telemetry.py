import asyncio
import random
import time
import numpy as np
from telemetry_bridge import TelemetryBridge

async def run_simulation():
    bridge = TelemetryBridge()
    await bridge.connect()
    
    print(" STARTING ALIEN SIMULATION...")
    print("Press Ctrl+C to stop.")
    
    equity = 10000.0
    
    try:
        while True:
            # Simulate Market Moves
            change = (random.random() - 0.48) * 50 # Slight upward drift
            equity += change
            
            # Simulate Topology
            # Betti-0: Random noise around 100
            b0 = [random.random() for _ in range(int(100 + random.random() * 50))]
            # Betti-1: Random noise around 20
            b1 = [random.random() for _ in range(int(20 + random.random() * 10))]
            
            telemetry = {
                "timestamp": time.time(),
                "pnl": round((equity - 10000) / 10000 * 100, 2),
                "equity": round(equity, 2),
                "drawdown": round(random.random() * 2, 2),
                "tti": 1.5 + random.random() * 1.5, # 1.5 to 3.0 (triggers alerts > 2.5)
                "positions": [
                    {
                        "symbol": "BTC-USDT",
                        "side": "LONG" if random.random() > 0.5 else "SHORT",
                        "size": round(random.random() * 2, 3),
                        "pnl": round((random.random() - 0.5) * 500, 2),
                        "leverage": 5
                    },
                    {
                        "symbol": "ETH-USDT",
                        "side": "LONG",
                        "size": round(random.random() * 10, 3),
                        "pnl": round((random.random() - 0.5) * 200, 2),
                        "leverage": 3
                    }
                ],
                "topology": {
                    "persistence_image": [], # Can be empty for now
                    "betti_curves": [b0, b1],
                    "wasserstein_dist": 0.04 + random.random() * 0.02
                }
            }
            
            await bridge.send(telemetry)
            print(f"Sent update: Equity=${equity:.2f} TTI={telemetry['tti']:.2f}")
            
            # 16ms update rate (60fps)
            await asyncio.sleep(0.05) 
            
    except KeyboardInterrupt:
        print("\nStopping simulation.")
        await bridge.close()

if __name__ == "__main__":
    asyncio.run(run_simulation())
