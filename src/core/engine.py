import time
from eth_account import Account
from src.config import Config
from src.core.logger import setup_logger
from src.core.monitor import Monitor
from src.data.feed import HyperliquidFeed
from src.alpha.signal_engine import SignalEngine
from src.risk.sizing import RiskManager
from src.execution.router import Router

logger = setup_logger("MEDALLION_ENGINE")

class Engine:
    def __init__(self):
        self.account = Account.from_key(Config.PRIVATE_KEY)
        self.feed = HyperliquidFeed(Config.API_URL)
        self.signal_engine = SignalEngine()
        self.risk_manager = RiskManager()
        self.signal_engine = SignalEngine()
        self.risk_manager = RiskManager()
        
        # Retry logic for API connection
        max_retries = 5
        for i in range(max_retries):
            try:
                self.router = Router(self.account, Config.API_URL)
                break
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    wait = 2 ** i
                    logger.warning(f"API RATE LIMIT (429). RETRYING IN {wait}s...")
                    time.sleep(wait)
                else:
                    raise e
        else:
            raise Exception("FAILED TO CONNECT TO API AFTER RETRIES")
            
        self.monitor = Monitor()
        
    def run(self):
        logger.info("ENGINE INITIALIZED. STARTING WARFARE LOOP.")
        
        while True:
            try:
                start_time = time.time()
                
                # 1. Update State
                user_state = self.feed.get_user_state(Config.WALLET_ADDRESS)
                if not user_state: continue
                
                equity = float(user_state['marginSummary']['accountValue'])
                logger.info(f"EQUITY: ${equity:.2f}")
                
                # 2. Scan Universe
                universe = self.feed.get_universe(Config.COIVERSE_SIZE)
                
                # Initialize variables for monitor update in case loop doesn't run
                price = None
                signal = 0
                atr = None
                narrative_id = None
                hurst = None
                fusion_score = None
                coin = None

                for coin_item in universe:
                    df = self.feed.get_candles(coin_item, Config.TIMEFRAME)
                    if df.empty: continue
                    
                    # 3. Alpha
                    # Pass feed and coin to analyze for Fusion Factor
                    signal, atr, narrative_id, hurst, fusion_score = self.signal_engine.analyze(df, self.feed, coin_item)
                    price = df.iloc[-1]['c']
                    coin = coin_item # Update coin for monitor

                    if signal != 0:
                        logger.info(f"NARRATIVE: {narrative_id} | HURST: {hurst:.2f} | FUSION: {fusion_score}")
                        # 4. Risk
                        size_usd = self.risk_manager.calculate_size(equity, price, atr)
                        
                        # 5. Execution
                        side = 'LONG' if signal == 1 else 'SHORT'
                        logger.info(f"SIGNAL {side} {coin_item}")
                        self.router.execute(coin_item, side, size_usd, price)
                        
                        # 6. Online Learning (Self-Correction)
                        # If we executed, we assume the signal was correct for now (Reinforcement later)
                        # For MVP, we train on the signal we just generated to reinforce it, 
                        # OR better: train on PAST signals if we had outcome data.
                        # Here: Auto-Encoder style reinforcement of own conviction (Hebbian).
                        target = 1 if signal == 1 else 2
                        loss = self.signal_engine.train_step(df, target)
                        logger.info(f"TRAINING STEP {coin_item} | LOSS: {loss:.4f}")
                
                # Periodic Save
                if int(time.time()) % 300 == 0: # Every 5 mins
                    self.signal_engine.save_model()
                    logger.info("BRAIN SAVED")

                
                elapsed = time.time() - start_time
                sleep_time = max(0, 60 - elapsed)
                
                # 6. Monitor Update
                # Prepare history for dashboard (last N candles)
                history = []
                if not df.empty:
                    # Take last N candles
                    recent = df.tail(Config.DASHBOARD_HISTORY_LIMIT).copy()
                    # Convert to list of dicts with timestamp (if we had it, but we only have OHLCV)
                    # We'll just send the values. 
                    # Actually, let's send columns: c, o, h, l, v
                    history = recent.to_dict('records')

                state = {
                    "equity": equity,
                    "price": price,
                    "signal": signal,
                    "atr": atr,
                    "narrative": narrative_id,
                    "hurst": hurst,
                    "fusion": fusion_score,
                    "coin": coin,
                    "history": history
                }
                self.monitor.update(state)
                
                logger.info(f"CYCLE COMPLETE. SLEEPING {sleep_time:.2f}s")
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("SHUTDOWN REQUESTED.")
                break
            except Exception as e:
                logger.error(f"CRITICAL LOOP FAILURE: {e}")
                time.sleep(10)
