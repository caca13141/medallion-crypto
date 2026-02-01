import asyncio
import sys
import os
import logging

sys.path.append(os.getcwd())

from src.execution.telemetry_bridge import TelemetryBridge

async def main():
    print("Testing Bridge...")
    bridge = TelemetryBridge()
    print("Connecting...")
    await bridge.connect()
    print("Connected!")
    await bridge.send({"test": "data"})
    print("Sent!")
    await bridge.close()

if __name__ == "__main__":
    asyncio.run(main())
