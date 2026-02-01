"""
CHAOS MONKEY - FAULT INJECTION & STRESS TESTING
Simulates production failures to test system resilience:
- Missing data (API failures)
- Delayed data (network latency)
- Corrupted topology (numerical instabilities)
"""

import numpy as np
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import random

@dataclass
class StressTestResult:
    """Results from a chaos stress test."""
    test_name: str
    total_iterations: int
    failures: int
    recoveries: int
    avg_recovery_time_ms: float
    accuracy_degradation: float  # % drop in prediction accuracy
    crash_rate: float

class ChaosMonkey:
    """
    Injects faults into the data pipeline to test resilience.
    """
    def __init__(self, failure_rate=0.1):
        """
        Args:
            failure_rate: Probability of injecting a fault per iteration (0.0-1.0)
        """
        self.failure_rate = failure_rate
        self.fault_history = deque(maxlen=1000)
        
    def inject_missing_data(self, data: np.ndarray, missing_rate=0.2) -> np.ndarray:
        """
        Randomly replace data points with NaN values.
        
        Args:
            data: Input array
            missing_rate: Fraction of data to corrupt
            
        Returns:
            Array with NaN injections
        """
        corrupted = data.copy()
        n_corrupt = int(len(data) * missing_rate)
        corrupt_indices = np.random.choice(len(data), size=n_corrupt, replace=False)
        corrupted[corrupt_indices] = np.nan
        
        self.fault_history.append({
            'type': 'missing_data',
            'indices': corrupt_indices.tolist()
        })
        
        return corrupted
    
    def inject_api_delay(self, delay_ms_range=(100, 2000)):
        """
        Simulate network/API latency.
        
        Args:
            delay_ms_range: (min, max) delay in milliseconds
        """
        delay_ms = random.uniform(*delay_ms_range)
        time.sleep(delay_ms / 1000.0)
        
        self.fault_history.append({
            'type': 'api_delay',
            'delay_ms': delay_ms
        })
        
        return delay_ms
    
    def inject_corrupted_topology(self, 
                                  topology_signature,
                                  corruption_strength=0.3):
        """
        Perturb topology features with random noise.
        
        Args:
            topology_signature: TopologySignature object
            corruption_strength: Scale of corruption (0.0-1.0)
            
        Returns:
            Corrupted topology signature
        """
        # Add noise to numeric fields
        if hasattr(topology_signature, 'loop_score'):
            topology_signature.loop_score *= (1 + np.random.randn() * corruption_strength)
        
        if hasattr(topology_signature, 'tti'):
            topology_signature.tti *= (1 + np.random.randn() * corruption_strength)
        
        if hasattr(topology_signature, 'wasserstein_amp'):
            topology_signature.wasserstein_amp *= (1 + np.random.randn() * corruption_strength)
        
        # Corrupt persistence image
        if hasattr(topology_signature, 'persistence_image'):
            noise = np.random.randn(*topology_signature.persistence_image.shape) * corruption_strength
            topology_signature.persistence_image += noise
            topology_signature.persistence_image = np.maximum(topology_signature.persistence_image, 0)
        
        self.fault_history.append({
            'type': 'corrupted_topology',
            'strength': corruption_strength
        })
        
        return topology_signature
    
    def inject_price_spike(self, prices: np.ndarray, spike_magnitude=0.1) -> np.ndarray:
        """
        Inject a sudden price spike/flash crash.
        
        Args:
            prices: Price array
            spike_magnitude: % magnitude of spike (e.g., 0.1 = 10%)
            
        Returns:
            Prices with injected spike
        """
        spiked = prices.copy()
        spike_idx = random.randint(0, len(prices) - 1)
        spike_direction = random.choice([-1, 1])
        
        spiked[spike_idx] *= (1 + spike_direction * spike_magnitude)
        
        self.fault_history.append({
            'type': 'price_spike',
            'index': spike_idx,
            'magnitude': spike_magnitude * spike_direction
        })
        
        return spiked
    
    def should_inject_fault(self) -> bool:
        """Randomly decide whether to inject a fault this iteration."""
        return random.random() < self.failure_rate
    
    def run_stress_test(self,
                       pipeline_fn,
                       n_iterations=100,
                       fault_types=['missing_data', 'api_delay', 'corrupted_topology']) -> StressTestResult:
        """
        Run a stress test on a pipeline function.
        
        Args:
            pipeline_fn: Function to test (should return success/failure)
            n_iterations: Number of test iterations
            fault_types: Types of faults to inject
            
        Returns:
            StressTestResult with metrics
        """
        print(f" Starting Chaos Monkey Stress Test")
        print(f"   Iterations: {n_iterations}")
        print(f"   Failure Rate: {self.failure_rate * 100:.1f}%")
        print(f"   Fault Types: {fault_types}")
        
        failures = 0
        recoveries = 0
        recovery_times = []
        accuracy_before = []
        accuracy_after = []
        crashes = 0
        
        for i in range(n_iterations):
            try:
                # Decide whether to inject fault
                inject_fault = self.should_inject_fault()
                
                if inject_fault:
                    fault_type = random.choice(fault_types)
                    
                    # Measure recovery time
                    start_time = time.time()
                    
                    # Run pipeline with fault
                    result = pipeline_fn(fault=fault_type, chaos_monkey=self)
                    
                    recovery_time_ms = (time.time() - start_time) * 1000
                    recovery_times.append(recovery_time_ms)
                    
                    if result.get('success', False):
                        recoveries += 1
                        accuracy_after.append(result.get('accuracy', 0.5))
                    else:
                        failures += 1
                else:
                    # Normal execution
                    result = pipeline_fn(fault=None, chaos_monkey=self)
                    
                    if result.get('success', True):
                        accuracy_before.append(result.get('accuracy', 0.5))
                
            except Exception as e:
                print(f"    Crash on iteration {i}: {e}")
                crashes += 1
                failures += 1
            
            if (i + 1) % 10 == 0:
                print(f"   Iteration {i + 1}/{n_iterations} - Failures: {failures}, Recoveries: {recoveries}")
        
        # Compute metrics
        avg_recovery_time = np.mean(recovery_times) if recovery_times else 0.0
        
        baseline_accuracy = np.mean(accuracy_before) if accuracy_before else 0.5
        fault_accuracy = np.mean(accuracy_after) if accuracy_after else 0.5
        accuracy_degradation = (baseline_accuracy - fault_accuracy) / baseline_accuracy if baseline_accuracy > 0 else 0.0
        
        crash_rate = crashes / n_iterations
        
        return StressTestResult(
            test_name="Chaos Monkey Stress Test",
            total_iterations=n_iterations,
            failures=failures,
            recoveries=recoveries,
            avg_recovery_time_ms=avg_recovery_time,
            accuracy_degradation=accuracy_degradation,
            crash_rate=crash_rate
        )

# Example Usage
if __name__ == "__main__":
    chaos = ChaosMonkey(failure_rate=0.2)
    
    # Mock pipeline function
    def mock_pipeline(fault=None, chaos_monkey=None):
        """Simulated pipeline for testing."""
        if fault == 'missing_data':
            # Simulate data with missing values
            data = np.random.randn(100)
            data = chaos_monkey.inject_missing_data(data, missing_rate=0.2)
            
            # Try to process (fillna)
            data = np.nan_to_num(data, nan=np.nanmean(data))
            
            return {'success': True, 'accuracy': 0.6}
        
        elif fault == 'api_delay':
            # Simulate delay
            delay_ms = chaos_monkey.inject_api_delay()
            
            # Recovery: timeout after 1s
            if delay_ms > 1000:
                return {'success': False, 'accuracy': 0.0}
            else:
                return {'success': True, 'accuracy': 0.7}
        
        elif fault == 'corrupted_topology':
            # Simulate corrupted topology
            class MockTopo:
                loop_score = 2.5
                tti = 1.5
                wasserstein_amp = 0.8
                persistence_image = np.random.rand(20, 20)
            
            topo = MockTopo()
            topo = chaos_monkey.inject_corrupted_topology(topo, corruption_strength=0.3)
            
            # Use corrupted features (degraded accuracy)
            return {'success': True, 'accuracy': 0.55}
        
        else:
            # Normal execution
            return {'success': True, 'accuracy': 0.8}
    
    # Run stress test
    result = chaos.run_stress_test(mock_pipeline, n_iterations=100)
    
    print("\n" + "="*60)
    print("CHAOS MONKEY RESULTS")
    print("="*60)
    print(f"Total Iterations: {result.total_iterations}")
    print(f"Failures: {result.failures}")
    print(f"Recoveries: {result.recoveries}")
    print(f"Avg Recovery Time: {result.avg_recovery_time_ms:.1f}ms")
    print(f"Accuracy Degradation: {result.accuracy_degradation * 100:.1f}%")
    print(f"Crash Rate: {result.crash_rate * 100:.1f}%")
    
    if result.crash_rate < 0.05 and result.accuracy_degradation < 0.2:
        print("\n SYSTEM IS RESILIENT")
    else:
        print("\n SYSTEM NEEDS HARDENING")
