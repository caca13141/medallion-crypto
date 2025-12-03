"""
TopoOmega v2.0 Test Suite
Verifies topology pipeline functionality
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from src.topology.integrator import TopologyIntegrator
from src.forecasting.topo_transformer import TopologicalTransformer
from src.alpha.topo_signal_engine import TopoSignalEngine
import torch

def generate_synthetic_data(n=200, regime='trending'):
    """Generate synthetic OHLCV data for testing"""
    np.random.seed(42)
    
    if regime == 'trending':
        # Uptrend with noise
        trend = np.linspace(100, 150, n)
        noise = np.random.randn(n) * 2
        close = trend + noise
    elif regime == 'ranging':
        # Mean-reverting range
        close = 100 + 10 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.randn(n)
    else:
        # Random walk
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    volume = np.random.uniform(1000, 5000, n)
    
    df = pd.DataFrame({
        'c': close,
        'h': high,
        'l': low,
        'v': volume
    })
    
    return df

def test_topology_integrator():
    """Test TopologyIntegrator"""
    print("\n" + "="*60)
    print("TEST 1: Topology Integrator")
    print("="*60)
    
    integrator = TopologyIntegrator(lookback=100, resolution=20)
    
    # Test on trending data
    df_trend = generate_synthetic_data(200, 'trending')
    result_trend = integrator.analyze(df_trend)
    
    print(f"Trending Market:")
    print(f"  Loop Score: {result_trend['loop_score']:.4f}")
    print(f"  TTI: {result_trend['tti']:.4f}")
    print(f"  Regime: {integrator.get_regime(result_trend['loop_score'], result_trend['tti'])}")
    print(f"  H0 Image Shape: {result_trend['persistence_image_h0'].shape}")
    print(f"  H1 Image Shape: {result_trend['persistence_image_h1'].shape}")
    
    # Test on ranging data
    df_range = generate_synthetic_data(200, 'ranging')
    result_range = integrator.analyze(df_range)
    
    print(f"\nRanging Market:")
    print(f"  Loop Score: {result_range['loop_score']:.4f}")
    print(f"  TTI: {result_range['tti']:.4f}")
    print(f"  Regime: {integrator.get_regime(result_range['loop_score'], result_range['tti'])}")
    
    assert result_trend['loop_score'] >= 0, "Loop score should be non-negative"
    assert result_trend['tti'] >= 0, "TTI should be non-negative"
    assert result_trend['persistence_image_h0'].shape == (20, 20), "H0 image wrong shape"
    
    print("✅ Topology Integrator PASSED")

def test_transformer_architecture():
    """Test TopologicalTransformer architecture"""
    print("\n" + "="*60)
    print("TEST 2: Topological Transformer")
    print("="*60)
    
    model = TopologicalTransformer(
        img_size=20,
        seq_len=72,
        d_model=256,
        nhead=8,
        num_layers=4  # Smaller for testing
    )
    
    # Create dummy input: (batch=2, seq_len=72, 1, 20, 20)
    batch_size = 2
    seq_len = 72
    img_seq = torch.randn(batch_size, seq_len, 1, 20, 20)
    
    # Forward pass
    time_logits, strength, confidence = model(img_seq)
    
    print(f"Input Shape: {img_seq.shape}")
    print(f"Time Logits Shape: {time_logits.shape} (should be [2, 48])")
    print(f"Strength Shape: {strength.shape} (should be [2, 1])")
    print(f"Confidence Shape: {confidence.shape} (should be [2, 1])")
    
    assert time_logits.shape == (batch_size, 48), "Time logits wrong shape"
    assert strength.shape == (batch_size, 1), "Strength wrong shape"
    assert confidence.shape == (batch_size, 1), "Confidence wrong shape"
    assert torch.all((strength >= 0) & (strength <= 1)), "Strength not in [0,1]"
    assert torch.all((confidence >= 0) & (confidence <= 1)), "Confidence not in [0,1]"
    
    print("✅ Transformer Architecture PASSED")

def test_signal_engine():
    """Test TopoSignalEngine"""
    print("\n" + "="*60)
    print("TEST 3: Topo Signal Engine")
    print("="*60)
    
    engine = TopoSignalEngine(
        enable_transformer=False,  # Skip transformer for quick test
        enable_ppo=False
    )
    
    df = generate_synthetic_data(200, 'trending')
    
    signal, leverage, confidence, metadata = engine.analyze(df)
    
    print(f"Signal: {signal} (-1=SHORT, 0=NEUTRAL, 1=LONG)")
    print(f"Leverage: {leverage:.1f}x")
    print(f"Confidence: {confidence:.2f}")
    print(f"Metadata:")
    print(f"  Loop Score: {metadata['loop_score']:.4f}")
    print(f"  TTI: {metadata['tti']:.4f}")
    print(f"  Regime: {metadata['regime']}")
    
    assert signal in [-1, 0, 1], "Signal must be -1, 0, or 1"
    assert 1 <= leverage <= 25, "Leverage must be in [1, 25]"
    assert 0 <= confidence <= 1, "Confidence must be in [0, 1]"
    
    print("✅ Signal Engine PASSED")

def test_risk_controls():
    """Test Nuclear Risk Controls"""
    print("\n" + "="*60)
    print("TEST 4: Nuclear Risk Controls")
    print("="*60)
    
    from src.risk.nuclear_controls import NuclearRiskControls
    
    controls = NuclearRiskControls(
        tti_threshold=3.0,
        confidence_min=0.6,
        daily_dd_limit=0.035
    )
    
    # Test TTI kill-switch
    should_flatten, reason = controls.should_flatten(
        tti=3.5, daily_equity=9000, start_equity=10000
    )
    print(f"High TTI (3.5): Flatten={should_flatten}, Reason={reason}")
    assert should_flatten, "Should flatten on high TTI"
    
    # Test daily DD kill-switch
    controls2 = NuclearRiskControls()
    should_flatten, reason = controls2.should_flatten(
        tti=1.0, daily_equity=9600, start_equity=10000
    )
    print(f"4% DD: Flatten={should_flatten}, Reason={reason}")
    assert should_flatten, "Should flatten on daily DD breach"
    
    # Test confidence-based leverage cap
    capped_lev = controls.calculate_leverage_cap(confidence=0.5, base_leverage=10)
    print(f"Low Confidence (0.5): Capped Leverage={capped_lev:.1f}x (should be 1x)")
    assert capped_lev == 1.0, "Should cap to 1x on low confidence"
    
    capped_lev = controls.calculate_leverage_cap(confidence=0.9, base_leverage=10)
    print(f"High Confidence (0.9): Capped Leverage={capped_lev:.1f}x")
    assert capped_lev > 1.0, "Should allow higher leverage on high confidence"
    
    print("✅ Risk Controls PASSED")

def run_all_tests():
    """Run complete test suite"""
    print("\n" + "="*60)
    print("🦅 TOPOOMEGA v2.0 TEST SUITE")
    print("="*60)
    
    try:
        test_topology_integrator()
        test_transformer_architecture()
        test_signal_engine()
        test_risk_controls()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED - TOPOOMEGA v2.0 OPERATIONAL")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
