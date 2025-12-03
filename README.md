# 🦅 MEDALLION CRYPTO TRADING BOT

**"The most profitable automated trading system in history."**

## Architecture
- **Core**: Event Loop, Logging, State Management.
- **Data**: Hyperliquid API (Feed), Historical Candles, Funding/OI.
- **Alpha**: 
    - **Transformer**: Topology Dissolution (Price Prediction).
    - **Narrative**: KeplerMapper + DBSCAN (Regime Detection).
    - **Persistence**: Hurst Exponent (Trend vs Mean Reversion).
    - **Fusion**: On-chain Funding/OI (Squeeze Detection).
    - **PPO**: Reinforcement Learning Agent (Optional).
- **Risk**: Kelly Criterion, Volatility Targeting (ATR), Max Drawdown Limit.
- **Execution**: Hyperliquid Router (Market/Limit Orders).

## Installation
1. **Prerequisites**: Python 3.12+, Hyperliquid Account.
2. **Setup**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Config**:
   - Edit `src/config.py` for parameters.
   - Set `WALLET_ADDRESS` and `PRIVATE_KEY` in `.env` (if live trading).

## Usage
**Launch the War Room (Engine + Dashboard):**
```bash
chmod +x launch.sh
./launch.sh
```

**Manual Run:**
```bash
# Terminal 1: Engine
python main.py

# Terminal 2: Dashboard
streamlit run dashboard.py
```

## Monitoring
- **Dashboard**: `http://localhost:8501`
- **Logs**: `engine.log` (created by launch.sh) or console output.

## Risk Management
- **Risk Per Trade**: 0.5% of Equity.
- **Max Leverage**: 1x (Spot-like safety).
- **Stop Loss**: Dynamic based on ATR.

## Disclaimer
Trading cryptocurrencies involves significant risk. This software is for educational purposes only. Use at your own risk.
