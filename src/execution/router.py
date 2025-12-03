from hyperliquid.exchange import Exchange
from eth_account.signers.local import LocalAccount
from src.core.logger import setup_logger

logger = setup_logger("EXECUTION")

class Router:
    def __init__(self, account: LocalAccount, api_url):
        self.exchange = Exchange(account, api_url, account_address=account.address)
        
    def execute(self, coin, side, size_usd, price):
        try:
            size_token = round(size_usd / price, 4)
            if size_token <= 0: return
            
            logger.info(f"ROUTING {side} {coin} | Size: {size_token} | Price: {price}")
            
            is_buy = (side == 'LONG')
            # Uncomment to enable live trading
            # result = self.exchange.market_open(coin, is_buy, size_token, price, 0.01)
            # logger.info(f"ORDER SENT: {result}")
            
        except Exception as e:
            logger.error(f"EXECUTION FAILURE {coin}: {e}")
