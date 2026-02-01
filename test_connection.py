import asyncio
import websockets
import json
import time

async def test_connection():
    uri = "ws://localhost:3000/ws"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected. Waiting for Marksman Pulse...")
            start_time = time.time()
            while time.time() - start_time < 30:
                message = await websocket.recv()
                data = json.loads(message)
                print(f"[{time.strftime('%H:%M:%S')}] Received Topic: {data.get('topic')}")
                if data.get('topic') == 'marksman_pulse':
                    payload = data.get('payload', {})
                    keys = list(payload.keys())
                    print(f"  Pulse Keys: {keys}")
                    if 'l3_book' in payload:
                        print("  SUCCESS: l3_book found in pulse.")
                        return
                else:
                    print(f"  (Filtered out internal noise: {data.get('topic')})")
            print("FAILED: No marksman_pulse received within 30s.")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
