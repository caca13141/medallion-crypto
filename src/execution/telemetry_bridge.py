import asyncio
import websockets
import json
import time
import logging

class TelemetryBridge:
    def __init__(self, uri="ws://127.0.0.1:9001"):
        self.uri = uri
        self.websocket = None
        self.logger = logging.getLogger("TelemetryBridge")
        logging.basicConfig(level=logging.INFO)

    async def connect(self):
        """Connect to the Rust WebSocket server."""
        while True:
            try:
                self.websocket = await websockets.connect(self.uri)
                self.logger.info(f" Connected to Dashboard at {self.uri}")
                return
            except Exception as e:
                self.logger.warning(f" Connection failed: {e}. Retrying in 2s...")
                await asyncio.sleep(2)

    async def send(self, data: dict):
        """Send telemetry data dictionary as JSON."""
        if not self.websocket:
            await self.connect()
        
        try:
            # Ensure timestamp is present
            if 'timestamp' not in data:
                data['timestamp'] = time.time()
                
            await self.websocket.send(json.dumps(data))
            # self.logger.debug("Sent telemetry update")
        except websockets.exceptions.ConnectionClosed:
            self.logger.error(" Connection lost. Reconnecting...")
            await self.connect()
            await self.websocket.send(json.dumps(data))
        except Exception as e:
            self.logger.error(f" Failed to send: {e}")

    async def close(self):
        if self.websocket:
            await self.websocket.close()

# Synchronous wrapper for non-async engines
class SyncTelemetryBridge:
    def __init__(self, uri="ws://127.0.0.1:9001"):
        self.bridge = TelemetryBridge(uri)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.bridge.connect())

    def send(self, data: dict):
        self.loop.run_until_complete(self.bridge.send(data))
