"""
Backtester: Walk-forward test on historical data
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import pickle
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Backtester:
    """Simple walk-forward backtester"""
    
    def __init__(self, model, initial_capital=10000):
        self.model = model
        self.initial_capital = initial_capital
        
    def backtest(self, X_test, y_test, prices_test):
        """
        Run backtest on test set.
        
        Args:
            X_test: features
            y_test: actual labels (not used for signals, just evaluation)
            prices_test: actual prices for PnL calculation
        """
        print("Running backtest...")
        
        # Get predictions
        predictions = self.model.predict(X_test)
        
        # Trading simulation
        equity = self.initial_capital
        position = 0  # -1, 0, 1
        entry_price = 0
        
        equity_curve = [equity]
        trades = []
        
        for i in range(len(predictions)):
            signal = predictions[i]
            current_price = prices_test[i]
            
            # Close position if signal changes
            if position != 0 and position != signal:
                # Calculate PnL
                if position == 1:
                    pnl = (current_price / entry_price - 1) * equity * 0.5  # 50% of equity
                else:  # position == -1
                    pnl = (entry_price / current_price - 1) * equity * 0.5
                
                # Apply fees (0.05% per side = 0.1% round trip)
                pnl -= equity * 0.5 * 0.001
                
                equity += pnl
                
                trades.append({
                    'entry': entry_price,
                    'exit': current_price,
                    'side': 'LONG' if position == 1 else 'SHORT',
                    'pnl': pnl,
                    'return': pnl / (equity - pnl)
                })
                
                position = 0
            
            # Open new position
            if signal != 0 and position == 0:
                position = signal
                entry_price = current_price
            
            equity_curve.append(equity)
        
        # Close final position
        if position != 0:
            current_price = prices_test[-1]
            if position == 1:
                pnl = (current_price / entry_price - 1) * equity * 0.5
            else:
                pnl = (entry_price / current_price - 1) * equity * 0.5
            pnl -= equity * 0.5 * 0.001
            equity += pnl
            
            trades.append({
                'entry': entry_price,
                'exit': current_price,
                'side': 'LONG' if position == 1 else 'SHORT',
                'pnl': pnl,
                'return': pnl / (equity - pnl)
            })
        
        # Calculate metrics
        final_equity = equity
        total_return = (final_equity / self.initial_capital - 1) * 100
        
        if len(trades) > 0:
            wins = sum(1 for t in trades if t['pnl'] > 0)
            losses = sum(1 for t in trades if t['pnl'] < 0)
            win_rate = wins / len(trades) if len(trades) > 0 else 0
            
            returns = np.array([t['return'] for t in trades])
            avg_return = np.mean(returns)
            sharpe = (avg_return / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Max drawdown
            peak = self.initial_capital
            max_dd = 0
            for eq in equity_curve:
                peak = max(peak, eq)
                dd = (peak - eq) / peak
                max_dd = max(max_dd, dd)
        else:
            win_rate = 0
            sharpe = 0
            max_dd = 0
            avg_return = 0
        
        metrics = {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return_pct': total_return,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd * 100,
            'avg_return_per_trade': avg_return * 100
        }
        
        return metrics, equity_curve, trades
    
    def plot_equity_curve(self, equity_curve, filename='results/equity_curve.png'):
        """Plot equity curve"""
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve, linewidth=2)
        plt.title('Equity Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Trade Number')
        plt.ylabel('Equity ($)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"✅ Equity curve saved to {filename}")

if __name__ == "__main__":
    # Load model
    with open('src/models/baseline_rf.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load test data
    with open('src/data/topology_dataset/fast_dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Load prices for test period
    df = pd.read_parquet('src/data/historical/btc_15m.parquet')
    
    # Calculate test period indices (matching dataset generation)
    lookback = 50
    forecast = 24
    test_start_idx = int((len(df) - lookback - forecast) * 0.8) + lookback
    test_end_idx = len(df) - forecast
    
    # Get exactly matching prices
    prices_test = df.iloc[test_start_idx:test_end_idx]['c'].values[:len(X_test)]
    
    # Backtest
    backtester = Backtester(model, initial_capital=10000)
    metrics, equity_curve, trades = backtester.backtest(X_test, y_test, prices_test)
    
    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    print("="*60)
    
    # Save
    with open('results/backtest_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    backtester.plot_equity_curve(equity_curve)
    
    print("\n✅ Backtest complete!")
