# 🦅 TOPOOMEGA v2.0 - NUCLEAR CRYPTO WARFARE

**"The most profitable automated trading system in history" - Upgraded**

## What Changed (v2.0 Upgrade)

### 🔬 Persistent Homology Core (Replaces Hurst Exponent)
- **GUDHI 3.11** + **Ripser 0.6.12** for real topological data analysis
- **Loop Score**: Measures H1 cycle strength (mean-reversion signal)
- **TTI (Topological Turbulence Index)**: Market chaos detector
- **Persistence Images**: Visual representations for ML input

### 🌊 Bifiltrated Persistence
- Multi-parameter analysis: **Correlation × Volume × Volatility**
- Zigzag persistence tracking
- Microstructure feature extraction

### 🧠 Topological Transformer
- Input: 72h of persistence images (not raw price)
- Output: 48h H1 loop dissolution forecast
- Target: >92% accuracy on dissolution timing

### 🚀 Wasserstein-PPO Agent
- Dynamic leverage: **1x-25x** based on topology + confidence
- Wasserstein auxiliary loss (predicted vs realized persistence)
- Replaces fixed Kelly sizing

### ⚠️ Nuclear Risk Controls
- **TTI Kill-Switch**: Auto-flatten on topological turbulence >3.0
- **Confidence Caps**: Low confidence → forced 1x leverage
- **Daily DD Hard Stop**: -3.5% → flatten + auto-restart
- **Leverage Scaling**: confidence-weighted (0.6-1.0 → 50%-100% of base)

### 📊 War Room Dashboard Upgrade
- Real-time topology metrics (Loop Score, TTI, Regime)
- Persistence diagram visualization
- Model confidence + dissolution forecasts
- Performance targets displayed

## Architecture (v2.0)

```
┌─────────────────────────────────────────────────────────┐
│                 TOPOOMEGA v2.0 PIPELINE                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  OHLCV Data → Topology Integrator                       │
│                  ├─ Persistence Engine (GUDHI/Ripser)   │
│                  ├─ Bifiltration Engine                 │
│                  └─ Persistence Images (20x20)          │
│                                                         │
│  Persistence Seq (72h) → Topological Transformer        │
│                           └─ H1 Dissolution Forecast    │
│                                                         │
│  Topo Features → Wasserstein-PPO Agent                  │
│                  └─ Position Size (1x-25x leverage)     │
│                                                         │
│  TTI + Confidence → Nuclear Risk Controls               │
│                     └─ Kill-Switch + Caps               │
│                                                         │
│  Output: Signal + Leverage + Metadata                   │
└─────────────────────────────────────────────────────────┘
```

## Installation

1. **Prerequisites**: Python 3.12+, Hyperliquid Account
2. **Setup**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Config**: Edit `src/config.py` and `.env`

## Usage

**TopoOmega v2.0 (Recommended):**
```bash
chmod +x topo_launch.sh
./topo_launch.sh
```

**Manual:**
```bash
# Terminal 1: Engine
python topo_omega_main.py

# Terminal 2: Dashboard
streamlit run topo_dashboard.py
```

**Test Suite:**
```bash
python tests/test_topoomega.py
```

## Target Performance

| Metric | Target | Method |
|--------|--------|--------|
| **CAGR** | 450-650% | Topology-guided leverage |
| **Sharpe** | 15-19 | Confidence-weighted sizing |
| **Max DD** | <7% | TTI kill-switch + DD limits |
| **Calmar** | >70 | CAGR / MaxDD |
| **Win Rate (Topo)** | >92% | H1 dissolution forecasts |

## Risk Management

- **Base Risk**: 0.5% per trade
- **Leverage Range**: 1x (safe) to 25x (high confidence)
- **TTI Threshold**: 3.0 (auto-flatten above)
- **Confidence Floor**: 0.6 (below = 1x only)
- **Daily DD Limit**: -3.5% (hard stop + auto-restart)

## Files (v2.0)

### New Topology Stack
- `src/topology/persistence_core.py` - Persistent homology engine
- `src/topology/bifiltration.py` - Multi-parameter filtration
- `src/topology/integrator.py` - Topology → Signal integration
- `src/forecasting/topo_transformer.py` - 72h→48h forecaster
- `src/rl/wasserstein_ppo.py` - Wasserstein-enhanced PPO
- `src/risk/nuclear_controls.py` - TTI + confidence controls
- `src/alpha/topo_signal_engine.py` - Main topology engine
- `src/core/topo_omega_engine.py` - Execution pipeline

### Entry Points
- `topo_omega_main.py` - Main launcher
- `topo_dashboard.py` - War room dashboard
- `topo_launch.sh` - Unified launch script

### Tests
- `tests/test_topoomega.py` - Topology pipeline tests

## Disclaimer

Trading cryptocurrencies involves significant risk. This software is for educational/research purposes. Use at your own risk.

**TopoOmega v2.0** - Nuclear Persistent Homology Warfare 🦅
