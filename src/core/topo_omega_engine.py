"""
Updated Engine: Integrates TopoOmega v2.0
"""
import time
from eth_account import Account
from src.config import Config
from src.core.logger import setup_logger
from src.core.monitor import Monitor
from src.data.feed import HyperliquidFeed
from src.alpha.topo_signal_engine import TopoSignalEngine
from src.risk.nuclear_controls import NuclearRiskControls
from src.execution.router import Router

logger = setup_logger("TOPO_OMEGA_ENGINE")

class TopoOmegaEngine:
    def __init__(self):
        # Account setup
        self.account = Account.from_key(Config.PRIVATE_KEY)
        
        # Data feed
        self.feed = HyperliquidFeed(Config.API_URL)
        
        # Nuclear signal engine
        self.signal_engine = TopoSignalEngine(
            enable_transformer=True,
            enable_ppo=False  # Enable after training
        )
        
        # Risk controls
        self.risk_controls = NuclearRiskControls(
            tti_threshold=3.0,
            confidence_min=0.6,
            daily_dd_limit=0.035,
            max_leverage=25
        )
        
        # Retry logic for API connection
        max_retries = 5
        for i in range(max_retries):
            try:
                self.router = Router(self.account, Config.API_URL)
                break
            except Exception as e:
                if "429" in str(e):
                    wait = 2 ** i
                    logger.warning(f"API RATE LIMIT. RETRYING IN {wait}s...")
                    time.sleep(wait)
                else:
                    raise e
        else:
            raise Exception("FAILED TO CONNECT TO API")
            
        self.monitor = Monitor()
        
        logger.info("TOPOOMEGA v2.0 ENGINE INITIALIZED")
        
    def run(self):
        logger.info("STARTING WARFARE LOOP...")
        
        # Reset daily equity at start
        state = self.feed.get_user_state(Config.WALLET_ADDRESS)
        if state:
            equity = float(state['marginSummary']['accountValue'])
            self.risk_controls.reset_daily(equity)
        
        while True:
            try:
                start_time = time.time()
                
                # 1. Account state
                state = self.feed.get_user_state(Config.WALLET_ADDRESS)
                if not state:
                    time.sleep(60)
                    continue
                    
                equity = float(state['marginSummary']['accountValue'])
                logger.info(f"EQUITY: ${equity:,.2f}")
                
                # 2. Scan universe
                universe = self.feed.get_universe(Config.COIVERSE_SIZE)
                
                # Process first coin for now (can expand to multi-coin)
                coin = universe[0] if universe else "BTC"
                
                # 3. Get data
                df = self.feed.get_candles(coin, Config.TIMEFRAME, limit=200)
                if df.empty:
                    time.sleep(60)
                    continue
                
                # 4. Topology analysis
                signal, leverage, confidence, metadata = self.signal_engine.analyze(
                    df, self.feed, coin
                )
                
                price = df.iloc[-1]['c']
                
                # 5. Log topology metrics
                logger.info(
                    f"{coin} | SIGNAL: {signal} | LEV: {leverage:.1f}x | "
                    f"CONF: {confidence:.2f} | TTI: {metadata.get('tti', 0):.2f} | "
                    f"REGIME: {metadata.get('regime', 'N/A')}"
                )
                
                # 6. Risk control check
                should_flatten, reason = self.risk_controls.should_flatten(
                    metadata.get('tti', 0),
                    equity,
                    self.risk_controls.daily_start_equity
                )
                
                if should_flatten:
                    logger.warning(f"FLATTEN TRIGGERED: {reason}")
                    signal = 0
                    leverage = 1
                
                # 7. Execution (if signal)
                if signal != 0:
                    # Calculate position size
                    atr = df['c'].rolling(14).std().iloc[-1]  # Simple volatility
                    size_usd = self.risk_controls.get_max_position_size(
                        equity, price, atr, confidence, leverage
                    )
                    
                    if size_usd > 0:
                        side = 'LONG' if signal == 1 else 'SHORT'
                        logger.info(f"EXECUTING {side} {coin} | SIZE: ${size_usd:.0f} | LEV: {leverage:.1f}x")
                        
                        # Execute
                        # self.router.execute(coin, side, size_usd, price)
                        
                # 8. Monitor update
                history = []
                if not df.empty:
                    recent = df.tail(Config.DASHBOARD_HISTORY_LIMIT).copy()
                    history = recent.to_dict('records')

                monitor_state = {
                    "equity": equity,
                    "price": price,
                    "signal": signal,
                    "leverage": leverage,
                    "confidence": confidence,
                    "tti": metadata.get('tti', 0),
                    "loop_score": metadata.get('loop_score', 0),
                    "regime": metadata.get('regime', 'N/A'),
                    "coin": coin,
                    "history": history
                }
                self.monitor.update(monitor_state)
                
                # 9. Model saving (every 5 minutes)
                if int(time.time()) % 300 == 0:
                    self.signal_engine.save_models()
                
                elapsed = time.time() - start_time
                sleep_time = max(0, 60 - elapsed)
                
                logger.info(f"CYCLE COMPLETE. SLEEPING {sleep_time:.2f}s")
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("SHUTDOWN INITIATED")
                break
            except Exception as e:
                logger.error(f"CYCLE ERROR: {e}")
                time.sleep(60)
