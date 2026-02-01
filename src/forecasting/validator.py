"""
Elite Forecasting Validator (RenTech-Grade)
Walk-forward validation with Information Coefficient tracking.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import spearmanr
import json
from pathlib import Path

@dataclass
class ValidationMetrics:
    """Validation metrics for a single fold"""
    ic: float  # Information Coefficient (Spearman rank correlation)
    ic_pvalue: float
    sharpe: float
    hit_rate: float  # Directional accuracy
    rmse: float
    mae: float
    turnover: float
    fold_id: int
    train_end: int
    test_end: int

class ForecastValidator:
    """
    Walk-Forward Validation Engine
    
    Key Metrics:
    - Information Coefficient (IC): Rank correlation between forecast and realized returns
    - Signal Decay: How IC degrades over forecast horizons
    - Out-of-Sample Sharpe: Risk-adjusted returns on unseen data
    """
    
    def __init__(self, 
                 train_window_hours: int = 168,  # 7 days
                 test_window_hours: int = 24,    # 1 day
                 step_size_hours: int = 24,      # Daily rolling
                 forecast_horizons: List[int] = [1, 4, 12, 24]):  # Hours
        
        self.train_window = train_window_hours
        self.test_window = test_window_hours
        self.step_size = step_size_hours
        self.horizons = forecast_horizons
        
        self.results: List[ValidationMetrics] = []
        self.ic_by_horizon: Dict[int, List[float]] = {h: [] for h in forecast_horizons}
        
    def compute_ic(self, 
                   forecasts: np.ndarray, 
                   actuals: np.ndarray) -> Tuple[float, float]:
        """
        Compute Information Coefficient (Spearman rank correlation)
        
        Args:
            forecasts: Model predictions
            actuals: Realized values
            
        Returns:
            (ic, p_value)
        """
        # Remove NaNs
        mask = ~(np.isnan(forecasts) | np.isnan(actuals))
        if mask.sum() < 10:
            return 0.0, 1.0
            
        ic, pval = spearmanr(forecasts[mask], actuals[mask])
        return float(ic), float(pval)
    
    def compute_hit_rate(self, 
                         forecasts: np.ndarray, 
                         actuals: np.ndarray) -> float:
        """Directional accuracy (% of correct sign predictions)"""
        mask = ~(np.isnan(forecasts) | np.isnan(actuals))
        if mask.sum() == 0:
            return 0.0
            
        correct = np.sign(forecasts[mask]) == np.sign(actuals[mask])
        return float(correct.mean())
    
    def compute_sharpe(self, returns: np.ndarray) -> float:
        """Annualized Sharpe ratio"""
        if len(returns) == 0 or np.isnan(returns).all():
            return 0.0
            
        clean_returns = returns[~np.isnan(returns)]
        if len(clean_returns) == 0:
            return 0.0
            
        mean_ret = clean_returns.mean()
        std_ret = clean_returns.std()
        
        if std_ret == 0:
            return 0.0
            
        # Annualize (assuming hourly returns)
        hourly_sharpe = mean_ret / std_ret
        annual_sharpe = hourly_sharpe * np.sqrt(24 * 365)
        
        return float(annual_sharpe)
    
    def compute_turnover(self, 
                        positions: np.ndarray) -> float:
        """Average daily position change (proxy for transaction costs)"""
        if len(positions) < 2:
            return 0.0
            
        changes = np.abs(np.diff(positions))
        return float(changes.mean())
    
    def walk_forward_split(self, 
                          data_length: int) -> List[Tuple[int, int, int, int]]:
        """
        Generate (train_start, train_end, test_start, test_end) indices
        
        Returns:
            List of tuples (train_start, train_end, test_start, test_end)
        """
        splits = []
        
        current_end = self.train_window + self.test_window
        
        while current_end <= data_length:
            train_start = current_end - self.train_window - self.test_window
            train_end = current_end - self.test_window
            test_start = train_end
            test_end = current_end
            
            splits.append((train_start, train_end, test_start, test_end))
            current_end += self.step_size
            
        return splits
    
    def validate_fold(self,
                     forecasts: np.ndarray,
                     actuals: np.ndarray,
                     fold_id: int,
                     train_end: int,
                     test_end: int) -> ValidationMetrics:
        """
        Validate a single fold
        
        Args:
            forecasts: Model predictions for test period
            actuals: Actual values for test period
            fold_id: Fold index
            train_end: End index of training period
            test_end: End index of test period
            
        Returns:
            ValidationMetrics object
        """
        # IC
        ic, ic_pval = self.compute_ic(forecasts, actuals)
        
        # Hit rate
        hit_rate = self.compute_hit_rate(forecasts, actuals)
        
        # Error metrics
        mask = ~(np.isnan(forecasts) | np.isnan(actuals))
        if mask.sum() > 0:
            rmse = float(np.sqrt(np.mean((forecasts[mask] - actuals[mask])**2)))
            mae = float(np.mean(np.abs(forecasts[mask] - actuals[mask])))
        else:
            rmse = mae = float('inf')
        
        # Sharpe (based on forecast-weighted returns)
        # Assume forecasts are returns, actuals are realized returns
        if mask.sum() > 0:
            strategy_returns = np.sign(forecasts[mask]) * actuals[mask]
            sharpe = self.compute_sharpe(strategy_returns)
        else:
            sharpe = 0.0
        
        # Turnover
        positions = np.sign(forecasts)
        turnover = self.compute_turnover(positions)
        
        metrics = ValidationMetrics(
            ic=ic,
            ic_pvalue=ic_pval,
            sharpe=sharpe,
            hit_rate=hit_rate,
            rmse=rmse,
            mae=mae,
            turnover=turnover,
            fold_id=fold_id,
            train_end=train_end,
            test_end=test_end
        )
        
        self.results.append(metrics)
        return metrics
    
    def analyze_signal_decay(self,
                            model,
                            data: np.ndarray,
                            targets_by_horizon: Dict[int, np.ndarray]) -> Dict[int, float]:
        """
        Analyze how IC degrades over forecast horizons
        
        Args:
            model: Trained forecasting model with .predict() method
            data: Input features
            targets_by_horizon: Dict mapping horizon (hours) to target values
            
        Returns:
            Dict mapping horizon to IC
        """
        decay_ics = {}
        
        for horizon in self.horizons:
            if horizon not in targets_by_horizon:
                continue
                
            # Generate forecasts
            forecasts = model.predict(data)
            actuals = targets_by_horizon[horizon]
            
            # Compute IC
            ic, _ = self.compute_ic(forecasts, actuals)
            decay_ics[horizon] = ic
            self.ic_by_horizon[horizon].append(ic)
            
        return decay_ics
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics across all folds"""
        if not self.results:
            return {}
            
        ics = [r.ic for r in self.results]
        sharpes = [r.sharpe for r in self.results if np.isfinite(r.sharpe)]
        hit_rates = [r.hit_rate for r in self.results]
        rmses = [r.rmse for r in self.results if np.isfinite(r.rmse)]
        maes = [r.mae for r in self.results if np.isfinite(r.mae)]
        
        # Compute half-life (when IC drops to 50% of initial)
        half_lives = {}
        for horizon in self.horizons:
            ic_series = self.ic_by_horizon.get(horizon, [])
            if len(ic_series) > 0:
                initial_ic = np.mean(ic_series[:10]) if len(ic_series) >= 10 else ic_series[0]
                half_lives[f'{horizon}h'] = initial_ic * 0.5
        
        return {
            'mean_ic': float(np.mean(ics)),
            'std_ic': float(np.std(ics)),
            'mean_sharpe': float(np.mean(sharpes)) if sharpes else 0.0,
            'mean_hit_rate': float(np.mean(hit_rates)),
            'mean_rmse': float(np.mean(rmses)) if rmses else float('inf'),
            'mean_mae': float(np.mean(maes)) if maes else float('inf'),
            'num_folds': len(self.results),
            'ic_by_horizon': {f'{h}h': float(np.mean(self.ic_by_horizon[h])) 
                             for h in self.horizons if self.ic_by_horizon[h]},
            'signal_half_lives': half_lives
        }
    
    def save_results(self, filepath: str):
        """Save validation results to JSON"""
        summary = self.get_summary_stats()
        
        # Convert results to dict
        results_dict = {
            'summary': summary,
            'folds': [
                {
                    'fold_id': r.fold_id,
                    'ic': r.ic,
                    'ic_pvalue': r.ic_pvalue,
                    'sharpe': r.sharpe if np.isfinite(r.sharpe) else None,
                    'hit_rate': r.hit_rate,
                    'rmse': r.rmse if np.isfinite(r.rmse) else None,
                    'mae': r.mae if np.isfinite(r.mae) else None,
                    'turnover': r.turnover,
                    'train_end': r.train_end,
                    'test_end': r.test_end
                }
                for r in self.results
            ]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f" Validation results saved to {filepath}")
    
    def should_retrain(self, 
                      recent_window: int = 10,
                      ic_threshold: float = 0.15) -> bool:
        """
        Determine if model should be retrained
        
        Criteria:
        - Recent IC (last N folds) < threshold
        - Significant IC degradation vs historical mean
        """
        if len(self.results) < recent_window:
            return False
            
        recent_ics = [r.ic for r in self.results[-recent_window:]]
        recent_mean_ic = np.mean(recent_ics)
        
        # Check if recent IC is below threshold
        if recent_mean_ic < ic_threshold:
            return True
            
        # Check if recent IC is significantly worse than historical
        if len(self.results) > recent_window * 2:
            historical_ics = [r.ic for r in self.results[:-recent_window]]
            historical_mean = np.mean(historical_ics)
            
            # If recent IC is 30% worse than historical
            if recent_mean_ic < historical_mean * 0.7:
                return True
                
        return False


if __name__ == "__main__":
    # Test the validator
    print(" Testing ForecastValidator...")
    
    validator = ForecastValidator(
        train_window_hours=168,
        test_window_hours=24,
        step_size_hours=24
    )
    
    # Generate synthetic data
    np.random.seed(42)
    data_length = 500
    
    # Mock forecasts and actuals with some correlation
    forecasts = np.random.randn(data_length)
    actuals = forecasts * 0.3 + np.random.randn(data_length) * 0.7
    
    # Test IC computation
    ic, pval = validator.compute_ic(forecasts[:100], actuals[:100])
    print(f"IC: {ic:.3f}, p-value: {pval:.4f}")
    
    # Test walk-forward splits
    splits = validator.walk_forward_split(data_length)
    print(f"Generated {len(splits)} folds")
    
    # Test validation
    for i, (train_start, train_end, test_start, test_end) in enumerate(splits[:5]):
        metrics = validator.validate_fold(
            forecasts[test_start:test_end],
            actuals[test_start:test_end],
            fold_id=i,
            train_end=train_end,
            test_end=test_end
        )
        print(f"Fold {i}: IC={metrics.ic:.3f}, Sharpe={metrics.sharpe:.2f}, Hit Rate={metrics.hit_rate:.2%}")
    
    # Get summary
    summary = validator.get_summary_stats()
    print("\n Summary Statistics:")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        else:
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Test retrain trigger
    should_retrain = validator.should_retrain()
    print(f"\nShould retrain: {should_retrain}")
