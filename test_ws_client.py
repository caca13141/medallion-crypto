
import asyncio
import aiohttp
import json

async def main():
    session = aiohttp.ClientSession()
    try:
        async with session.ws_connect('http://localhost:3000/ws') as ws:
            print("Connected to WebSocket Server")
            
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    print(f"Received Topic: {data.get('topic')}")
                    if data.get('topic') == 'marksman_pulse':
                        payload = data.get('payload', {})
                        keys = list(payload.keys())
                        print(f"  > Payload Keys: {keys}")
                        if 'l3_book' in keys:
                             book = payload['l3_book']
                             print(f"  > L3 Size: {len(book.get('bids', []))} bids")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print('ws connection closed with exception %s',
                          ws.exception())
    except Exception as e:
        print(f"Connection Failed: {e}")
    finally:
        await session.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
