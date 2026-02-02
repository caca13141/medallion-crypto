#!/bin/bash

# Trap SIGINT to kill background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

echo "=================================================="
echo "🚀 LAUNCHING JANE STREET POLYGLOT STACK"
echo "=================================================="

# 1. C++ Pricing Engine
echo "[1/3] Launching C++ Pricing Engine..."
./src/pricing_engine/build/pricing_engine &
CPP_PID=$!
sleep 1 # Let it initialize

# 2. Rust Execution Daemon
echo "[2/3] Launching Rust Execution Daemon..."
cd src/execution/rust_daemon 
cargo run --bin daemon --quiet &
RUST_PID=$!
cd ../../..
sleep 2 # Let it compile/start

# 3. OCaml Strategy Kernel
echo "[3/3] Launching OCaml Strategy Kernel..."
# Ensure opam env is set
eval $(opam env --switch=/Users/raphaelmaksoud/ocaml --set-switch)
cd src/strategy_kernel
dune exec ./bin/main.exe &
OCAML_PID=$!
cd ../../..

echo "=================================================="
echo "✅ SYSTEM ACTIVE: C++ (PID $CPP_PID) | Rust (PID $RUST_PID) | OCaml (PID $OCAML_PID)"
echo "Press Ctrl+C to shutdown all systems."
echo "=================================================="

# Wait for all processes
wait
