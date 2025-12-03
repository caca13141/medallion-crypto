import os
from dotenv import load_dotenv
from hyperliquid.utils import constants

load_dotenv()

class Config:
    WALLET_ADDRESS = os.getenv("WALLET_ADDRESS")
    PRIVATE_KEY = os.getenv("PRIVATE_KEY")
    API_URL = constants.MAINNET_API_URL
    
    # Trading Parameters
    COIVERSE_SIZE = 30
    TIMEFRAME = '15m'
    LOOKBACK = 100
    
    # Strategy Parameters
    EMA_FAST = 8
    EMA_SLOW = 21
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    ZIGZAG_THRESHOLD = 0.02
    
    # Risk Parameters
    RISK_PER_TRADE = 0.005 # Optimized to 0.5% (Survival Mode)
    MAX_DRAWDOWN_LIMIT = 0.10
    LEVERAGE_LIMIT = 1 # Optimized to 1x (Spot-like safety)

    # Transformer Parameters
    TRANSFORMER_D_MODEL = 64
    TRANSFORMER_NHEAD = 4
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_INPUT_DIM = 5 # c, o, h, l, v
    
    # Model Persistence & Training
    MODEL_PATH = "src/data/brain.pth"
    LEARNING_RATE = 1e-4

    # TDA / KeplerMapper Parameters
    TDA_PROJECTION = [0, 1] # Project to 2D (e.g. t-SNE or PCA components)
    TDA_RESOLUTION = 10
    TDA_GAIN = 0.1

    # Persistence Parameters
    HURST_THRESHOLD = 0.55 # > 0.55 implies strong trend

    # PPO (Reinforcement Learning) Parameters
    PPO_LEARNING_RATE = 3e-4
    PPO_GAMMA = 0.99
    PPO_GAE_LAMBDA = 0.95
    PPO_CLIP_RANGE = 0.2
    PPO_ENABLE = False # Set to True to use PPO agent
    PPO_TRAIN_TIMESTEPS = 100000


    # Loss Parameters
    WASSERSTEIN_GAIN = 0.5

    # Fusion Parameters
    FUSION_ENABLE = True
    FUNDING_THRESHOLD = 0.0001 # 0.01% per hour
    OI_ZSCORE_THRESHOLD = 2.0 # 2 std devs above mean

    # Dashboard Parameters
    DASHBOARD_HISTORY_LIMIT = 50








    
    @classmethod
    def validate(cls):
        if not cls.WALLET_ADDRESS or not cls.PRIVATE_KEY:
            raise ValueError("Missing Credentials in .env")
