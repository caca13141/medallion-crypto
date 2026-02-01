
import asyncio
import ccxt.async_support as ccxt
import os
from dotenv import load_dotenv

load_dotenv()

async def check():
    exchange = ccxt.hyperliquid({
        'apiKey': os.getenv("HYPERLIQUID_API_KEY"),
        'secret': os.getenv("HYPERLIQUID_SECRET"),
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })
    
    try:
        print("Fetching markets...")
        markets = await exchange.load_markets()
        print(f"Total markets: {len(markets)}")
        
        # Look for BTC
        btc_markets = [s for s in markets.keys() if 'BTC' in s]
        print(f"BTC Markets: {btc_markets}")
        
        target = "BTC/USDC:USDC"
        if target in markets:
            print(f"Target {target} FOUND. Fetching orderbook...")
            ob = await exchange.fetch_order_book(target, limit=5)
            print(f"Price: {ob['bids'][0][0]}")
        else:
            print(f"Target {target} NOT FOUND.")
            # Try a common one
            common = "BTC/USDT:USDT" if "BTC/USDT:USDT" in markets else btc_markets[0]
            print(f"Trying {common}...")
            ob = await exchange.fetch_order_book(common, limit=5)
            print(f"Price for {common}: {ob['bids'][0][0]}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(check())
