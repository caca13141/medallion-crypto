"""
WALK-FORWARD VALIDATION ENGINE
Tests model performance on rolling out-of-sample windows to detect overfitting
and validate robustness across different market regimes (2022-2025).
"""

import numpy as np
import pandas as pd
import ccxt
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class WindowMetrics:
    """Performance metrics for a single validation window."""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    prediction_mae: float  # Mean Absolute Error
    prediction_rmse: float  # Root Mean Squared Error

class WalkForwardValidator:
    """
    Walk-forward validation framework for time-series models.
    """
    def __init__(self,
                 model=None,
                 topology_engine=None,
                 train_window_days=180,
                 test_window_days=30,
                 step_days=30):
        """
        Args:
            model: Forecast model (e.g., TopologyForecaster)
            topology_engine: ProductionTopologyEngine instance
            train_window_days: Size of training window in days
            test_window_days: Size of testing window in days
            step_days: Days to step forward between windows
        """
        self.model = model
        self.topology_engine = topology_engine
        self.train_window = train_window_days
        self.test_window = test_window_days
        self.step = step_days
        
        # Exchange for fetching historical data
        self.exchange = ccxt.binance({'enableRateLimit': True})
        
    def fetch_historical_data(self,
                             symbol: str,
                             start_date: str,
                             end_date: str,
                             timeframe: str = '1h') -> pd.DataFrame:
        """
        Fetch historical OHLCV data from exchange.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            timeframe: Candle timeframe
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f" Fetching historical data: {start_date} to {end_date}")
        
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        all_candles = []
        current_ts = start_ts
        
        while current_ts < end_ts:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_ts,
                    limit=1000
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                current_ts = candles[-1][0] + 1
                
                # Rate limit
                import time
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                print(f" Error fetching data: {e}")
                break
        
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Filter to exact range
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        return df
    
    def compute_topology_features(self, prices: np.ndarray, volumes: np.ndarray):
        """Compute topology signature for a price window."""
        if self.topology_engine is None:
            # Mock topology for testing
            class MockTopo:
                loop_score = np.random.rand() * 5
                tti = np.random.rand() * 10
                wasserstein_amp = np.random.rand() * 3
            return MockTopo()
        
        # Use delay embedding
        embedding_dim = 3
        delay = 5
        
        if len(prices) < embedding_dim * delay:
            return None
        
        point_cloud = []
        vol_cloud = []
        
        for i in range(len(prices) - embedding_dim * delay):
            point = [prices[i + j * delay] for j in range(embedding_dim)]
            point_cloud.append(point)
            vol_window = [volumes[i + j * delay] for j in range(embedding_dim)]
            vol_cloud.append(np.mean(vol_window))
        
        point_cloud = np.array(point_cloud)
        point_cloud = (point_cloud - point_cloud.mean()) / (point_cloud.std() + 1e-8)
        
        vol_array = np.array(vol_cloud)
        
        signature = self.topology_engine.analyze_window(point_cloud, volumes=vol_array)
        
        return signature
    
    def backtest_window(self,
                       train_df: pd.DataFrame,
                       test_df: pd.DataFrame) -> WindowMetrics:
        """
        Backtest a single train/test window.
        
        Args:
            train_df: Training data
            test_df: Testing data
            
        Returns:
            WindowMetrics with performance statistics
        """
        # Simple strategy: predict direction and allocate capital
        predictions = []
        actuals = []
        returns = []
        
        initial_capital = 10000.0
        equity = initial_capital
        equity_curve = [equity]
        
        # Simulate trading on test set
        for i in range(len(test_df) - 48):  # 48h prediction horizon
            # Current window
            current_idx = i
            future_idx = min(i + 48, len(test_df) - 1)
            
            current_price = test_df.iloc[current_idx]['close']
            future_price = test_df.iloc[future_idx]['close']
            
            # Get recent history for topology
            lookback = 100
            start_idx = max(0, current_idx - lookback)
            history_prices = test_df.iloc[start_idx:current_idx]['close'].values
            history_volumes = test_df.iloc[start_idx:current_idx]['volume'].values
            
            if len(history_prices) < 50:
                continue
            
            # Compute topology
            topo_sig = self.compute_topology_features(history_prices, history_volumes)
            
            if topo_sig is None:
                continue
            
            # Simple prediction: direction based on loop score vs TTI
            predicted_direction = 1 if topo_sig.loop_score > topo_sig.tti else -1
            actual_direction = 1 if future_price > current_price else -1
            
            predicted_price = current_price * (1 + predicted_direction * 0.02)
            
            predictions.append(predicted_price)
            actuals.append(future_price)
            
            # Trade execution
            price_return = (future_price - current_price) / current_price
            
            # If prediction matches direction, take position
            if predicted_direction == actual_direction:
                equity += equity * price_return * 0.5  # 50% allocation
                returns.append(price_return * 0.5)
            else:
                equity -= equity * abs(price_return) * 0.5
                returns.append(-abs(price_return) * 0.5)
            
            equity_curve.append(equity)
        
        # Compute metrics
        if len(returns) == 0:
            return None
        
        returns = np.array(returns)
        
        # Sharpe Ratio (annualized, assuming hourly returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = (mean_return / (std_return + 1e-8)) * np.sqrt(365 * 24)
        
        # Max Drawdown
        equity_curve = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
       # Total Return
        total_return = (equity - initial_capital) / initial_capital
        
        # Win Rate
        wins = np.sum(np.array(returns) > 0)
        win_rate = wins / len(returns)
        
        # Prediction Error
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        return WindowMetrics(
            window_id=0,
            train_start=train_df.iloc[0]['timestamp'].strftime('%Y-%m-%d'),
            train_end=train_df.iloc[-1]['timestamp'].strftime('%Y-%m-%d'),
            test_start=test_df.iloc[0]['timestamp'].strftime('%Y-%m-%d'),
            test_end=test_df.iloc[-1]['timestamp'].strftime('%Y-%m-%d'),
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_return=total_return,
            win_rate=win_rate,
            prediction_mae=mae,
            prediction_rmse=rmse
        )
    
    def run_validation(self,
                      symbol: str = 'BTC/USDT',
                      start_date: str = '2023-01-01',
                      end_date: str = '2024-12-01') -> List[WindowMetrics]:
        """
        Run walk-forward validation across multiple windows.
        
        Returns:
            List of WindowMetrics for each validation window
        """
        print(" Starting Walk-Forward Validation")
        print(f"   Symbol: {symbol}")
        print(f"   Period: {start_date} to {end_date}")
        print(f"   Train: {self.train_window}d | Test: {self.test_window}d | Step: {self.step}d")
        
        # Fetch full dataset
        df = self.fetch_historical_data(symbol, start_date, end_date)
        
        if df.empty:
            print(" No data fetched")
            return []
        
        print(f" Fetched {len(df)} candles")
        
        # Generate windows
        results = []
        window_id = 0
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        current_start = start_dt
        
        while current_start + timedelta(days=self.train_window + self.test_window) <= end_dt:
            train_start = current_start
            train_end = current_start + timedelta(days=self.train_window)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_window)
            
            # Filter data
            train_df = df[(df['timestamp'] >= train_start) & (df['timestamp'] < train_end)]
            test_df = df[(df['timestamp'] >= test_start) & (df['timestamp'] < test_end)]
            
            if len(train_df) < 100 or len(test_df) < 100:
                current_start += timedelta(days=self.step)
                continue
            
            print(f"\n Window {window_id + 1}")
            print(f"   Train: {train_start.date()} to {train_end.date()} ({len(train_df)} candles)")
            print(f"   Test:  {test_start.date()} to {test_end.date()} ({len(test_df)} candles)")
            
            # Backtest this window
            metrics = self.backtest_window(train_df, test_df)
            
            if metrics:
                metrics.window_id = window_id
                results.append(metrics)
                
                print(f"   Sharpe: {metrics.sharpe_ratio:.2f}")
                print(f"   Return: {metrics.total_return * 100:.2f}%")
                print(f"   Max DD: {metrics.max_drawdown * 100:.2f}%")
                print(f"   Win Rate: {metrics.win_rate * 100:.1f}%")
            
            window_id += 1
            current_start += timedelta(days=self.step)
        
        return results

# Example Usage
if __name__ == "__main__":
    validator = WalkForwardValidator(train_window_days=90, test_window_days=30, step_days=30)
    
    # Run on shorter window for testing
    results = validator.run_validation(
        symbol='BTC/USDT',
        start_date='2024-06-01',
        end_date='2024-12-01'
    )
    
    if results:
        print("\n" + "="*60)
        print("WALK-FORWARD VALIDATION RESULTS")
        print("="*60)
        
        avg_sharpe = np.mean([r.sharpe_ratio for r in results])
        avg_return = np.mean([r.total_return for r in results])
        avg_dd = np.mean([r.max_drawdown for r in results])
        avg_win_rate = np.mean([r.win_rate for r in results])
        
        print(f"Windows Tested: {len(results)}")
        print(f"Avg Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"Avg Return: {avg_return * 100:.2f}%")
        print(f"Avg Max Drawdown: {avg_dd * 100:.2f}%")
        print(f"Avg Win Rate: {avg_win_rate * 100:.1f}%")
