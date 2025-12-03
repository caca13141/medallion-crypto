"""
TopoOmega v2.0 Main Entry Point
"""
import signal
import sys
from src.core.topo_omega_engine import TopoOmegaEngine

def signal_handler(sig, frame):
    print("\n🦅 TOPOOMEGA SHUTDOWN INITIATED")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 60)
    print("🦅 TOPOOMEGA v2.0 - NUCLEAR CRYPTO WARFARE")
    print("=" * 60)
    print("Persistent Homology | Bifiltrated Persistence | Wasserstein-PPO")
    print("Target: 450-650% CAGR | Sharpe 15-19 | MaxDD <7%")
    print("=" * 60)
    
    engine = TopoOmegaEngine()
    engine.run()

if __name__ == "__main__":
    main()
