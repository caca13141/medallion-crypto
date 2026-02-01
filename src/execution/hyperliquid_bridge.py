import ccxt.async_support as ccxt
import asyncio
import os
import json
import time
import numpy as np
from typing import Dict, Any, Optional

class HyperliquidBridge:
    """
    Elite Execution Bridge for Hyperliquid (DEX).
    Handles:
    - Wallet Connection (Private Key)
    - Order Placement (Limit/Market)
    - Position Monitoring
    - L1 Data Streaming
    """
    def __init__(self, wallet_address: str, private_key: str):
        self.wallet_address = wallet_address
        self.private_key = private_key
        self.exchange = ccxt.hyperliquid({
            'walletAddress': self.wallet_address,
            'privateKey': self.private_key,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',
                'fetchMarkets': {
                    'hip3': {
                        'dex': ['hyperliquid'] # Limit to official DEX
                    }
                }
            }
        })
        self.positions = {}
        self.orders = {}

    async def initialize(self):
        """Connect to exchange and load markets."""
        try:
            await self.exchange.load_markets()
        except Exception as e:
            print(f" load_markets failed ({e}). Injecting manual market data for BTC/USDC:USDC.")
            # Initialize markets dict if None
            if self.exchange.markets is None:
                self.exchange.markets = {}
            if self.exchange.markets_by_id is None:
                self.exchange.markets_by_id = {}
                
            # Manual injection for Hyperliquid Perp
            self.exchange.markets['BTC/USDC:USDC'] = {
                'id': 'BTC',
                'symbol': 'BTC/USDC:USDC',
                'base': 'BTC',
                'quote': 'USDC',
                'type': 'swap',
                'spot': False,
                'swap': True,
                'linear': True,
                'precision': {'amount': 0.0001, 'price': 0.1},
                'limits': {'amount': {'min': 0.0001}, 'cost': {'min': 5.0}},
            }
            self.exchange.markets_by_id['BTC'] = self.exchange.markets['BTC/USDC:USDC']
            
        print(f" Hyperliquid Bridge Initialized for {self.wallet_address[:6]}...")

    async def get_l1_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch Best Bid/Ask and recent trades."""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return {
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'last': ticker['last'],
                'timestamp': ticker['timestamp']
            }
        except Exception as e:
            print(f" fetch_ticker failed ({e}). Using MOCK L1 data.")
            # Fallback Mock Data
            return {
                'bid': 98000.0,
                'ask': 98001.0,
                'last': 98000.5,
                'timestamp': int(time.time() * 1000)
            }

    async def get_l2_snapshot(self, coin: str = "BTC") -> Dict[str, Any]:
        """
        Fetch L2 order book snapshot using official SDK (Info).
        Returns top 50 levels of bids and asks.
        """
        try:
            from hyperliquid.info import Info
            info = Info(skip_ws=True)
            l2_data = info.l2_snapshot(coin)
            
            if l2_data and 'levels' in l2_data:
                levels = l2_data['levels']
                bids = [[float(l['px']), float(l['sz'])] for l in levels[0][:50]]
                asks = [[float(l['px']), float(l['sz'])] for l in levels[1][:50]]
                
                return {
                    'bids': np.array(bids),
                    'asks': np.array(asks),
                    'mid': (bids[0][0] + asks[0][0]) / 2.0 if bids and asks else 0.0,
                    'timestamp': l2_data.get('time', int(time.time() * 1000))
                }
        except Exception as e:
            print(f" get_l2_snapshot failed ({e}). Returning empty.")
            
        return {'bids': np.array([]), 'asks': np.array([]), 'mid': 0.0, 'timestamp': 0}

    async def get_positions(self) -> Dict[str, Any]:
        """Fetch current open positions."""
        raw_positions = await self.exchange.fetch_positions()
        # Format for internal use
        self.positions = {
            p['symbol']: {
                'side': p['side'],
                'size': float(p['contracts']),
                'entry_price': float(p['entryPrice']),
                'pnl': float(p['unrealizedPnl']),
                'leverage': float(p['leverage'])
            }
            for p in raw_positions if float(p['contracts']) > 0
        }
        return self.positions

    async def place_order(self, symbol: str, side: str, size: float, price: Optional[float] = None, order_type: str = 'limit'):
        """
        Place an order on Hyperliquid.
        """
        try:
            if order_type == 'limit':
                order = await self.exchange.create_order(symbol, 'limit', side, size, price)
            else:
                order = await self.exchange.create_order(symbol, 'market', side, size)
            
            print(f" ORDER PLACED: {side} {size} {symbol} @ {price if price else 'MARKET'}")
            return order
        except Exception as e:
            print(f" ORDER FAILED: {e}")
            return None

    async def cancel_all_orders(self, symbol: str):
        """Cancel all open orders for a symbol."""
        try:
            await self.exchange.cancel_all_orders(symbol)
            print(f" Cancelled all orders for {symbol}")
        except Exception as e:
            print(f" Cancel Failed: {e}")

    async def close(self):
        await self.exchange.close()
