"""
Backtest Framework for Model Validation
Tests trained models on historical data with proper walk-forward methodology
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

class Backtester:
    """
    Walk-forward backtester for signal validation
    """
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.trades = []
        self.equity_curve = []
        
    def load_data(self, data_path):
        """Load test data"""
        df = pd.read_parquet(data_path)
        return df
    
    def compute_returns(self, signals, prices, fees=0.001):
        """
        Compute returns from signals
        
        Args:
            signals: Array of predicted signals (-1, 0, 1)
            prices: Array of prices
            fees: Trading fees (0.1% default)
        """
        # Shift signals by 1 (can't trade on current bar)
        signals = np.roll(signals, 1)
        signals[0] = 0
        
        # Compute returns
        price_returns = np.diff(prices) / prices[:-1]
        
        # Align arrays
        signals = signals[:-1]
        
        # Strategy returns = signal * price_return - fees on trades
        strategy_returns = signals * price_returns
        
        # Apply fees on position changes
        position_changes = np.abs(np.diff(np.concatenate([[0], signals])))
        strategy_returns -= position_changes * fees
        
        return strategy_returns
    
    def compute_metrics(self, returns):
        """Compute performance metrics"""
        
        # Annualized metrics (assuming 1-min bars, 525600 mins/year)
        periods_per_year = 525600
        
        metrics = {
            'total_return': np.prod(1 + returns) - 1,
            'sharpe': np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(periods_per_year),
            'sortino': np.mean(returns) / (np.std(returns[returns < 0]) + 1e-10) * np.sqrt(periods_per_year),
            'max_drawdown': self._compute_max_drawdown(returns),
            'win_rate': (returns > 0).sum() / len(returns),
            'avg_win': returns[returns > 0].mean() if (returns > 0).any() else 0,
            'avg_loss': returns[returns < 0].mean() if (returns < 0).any() else 0,
        }
        
        return metrics
    
    def _compute_max_drawdown(self, returns):
        """Compute maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def compute_ic(self, predictions, actual_returns):
        """
        Compute Information Coefficient (Spearman correlation)
        """
        # Align arrays
        min_len = min(len(predictions), len(actual_returns))
        predictions = predictions[:min_len]
        actual_returns = actual_returns[:min_len]
        
        # Remove NaNs
        mask = ~(np.isnan(predictions) | np.isnan(actual_returns))
        predictions = predictions[mask]
        actual_returns = actual_returns[mask]
        
        if len(predictions) < 10:
            return 0.0, 1.0
        
        ic, pvalue = spearmanr(predictions, actual_returns)
        return ic, pvalue
    
    def plot_results(self, returns, title="Backtest Results", output_path=None):
        """Plot backtest results"""
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # Equity curve
        equity = np.cumprod(1 + returns) * self.initial_capital
        axes[0].plot(equity, linewidth=1)
        axes[0].set_title(f'{title} - Equity Curve')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        axes[1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        axes[1].set_title('Drawdown')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True, alpha=0.3)
        
        # Returns distribution
        axes[2].hist(returns, bins=100, alpha=0.7, edgecolor='black')
        axes[2].axvline(0, color='red', linestyle='--', linewidth=1)
        axes[2].set_title('Returns Distribution')
        axes[2].set_xlabel('Return')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"   Plot saved to {output_path}")
        
        plt.close()

def backtest_regime_detector():
    """
    Backtest the Regime Detector model
    Strategy: Trade based on regime transitions
    """
    print("="*60)
    print("BACKTESTING REGIME DETECTOR")
    print("="*60)
    
    # Load model
    print("\n Loading trained model...")
    with open('src/data/models/hmm_model.pkl', 'rb') as f:
        checkpoint = pickle.load(f)
    
    model = checkpoint['model']
    scaler = checkpoint['scaler']
    
    # Load test data
    print(" Loading test data...")
    test_df = pd.read_parquet('data/splits/test_2025.parquet')
    
    # Prepare features
    import sys
    sys.path.append('src/training')
    from train_regime_detector import prepare_features
    X_test = prepare_features(test_df)
    
    # Predict regimes
    print(" Predicting regimes...")
    X_test_scaled = scaler.transform(X_test)
    regimes = model.predict(X_test_scaled)
    
    # Generate signals
    # Simple strategy: 
    # - State 3 (crisis): Short (-1)
    # - State 2 (low vol): Long (+1)
    # - Others: Neutral (0)
    signals = np.zeros(len(regimes))
    signals[regimes == 3] = -1  # Short crisis
    signals[regimes == 2] = 1   # Long low vol
    
    # Get prices
    prices = test_df['close'].values[:len(signals)]
    
    # Backtest
    print("\n Running backtest...")
    bt = Backtester(initial_capital=10000)
    returns = bt.compute_returns(signals, prices, fees=0.001)
    
    # Compute metrics
    metrics = bt.compute_metrics(returns)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
    print(f"Avg Win: {metrics['avg_win']*100:.4f}%")
    print(f"Avg Loss: {metrics['avg_loss']*100:.4f}%")
    
    # Information Coefficient
    price_returns = np.diff(prices) / prices[:-1]
    ic, pvalue = bt.compute_ic(signals[:-1], price_returns)
    print(f"\nInformation Coefficient: {ic:.4f} (p={pvalue:.4f})")
    
    # Plot
    bt.plot_results(returns, 
                    title="Regime Detector Backtest (2025)", 
                    output_path="data/backtest/regime_detector_results.png")
    
    # Save metrics
    results = {
        'model': 'regime_detector',
        'test_period': '2025',
        'metrics': metrics,
        'ic': ic,
        'pvalue': pvalue
    }
    
    output_path = Path('data/backtest/regime_detector_metrics.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n Results saved to {output_path}")
    
    return results

if __name__ == "__main__":
    results = backtest_regime_detector()
    
    print("\n" + "="*60)
    print(" BACKTEST COMPLETE")
    print("="*60)
    print(f"Sharpe: {results['metrics']['sharpe']:.2f}")
    print(f"IC: {results['ic']:.4f}")
    print("\nNext: Train Hawkes and test 2-model ensemble")
