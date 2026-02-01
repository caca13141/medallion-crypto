import torch
import torch.nn as nn
import time

print("Environment Check: Torch version", torch.__version__, flush=True)

# Minimal MoE-like model
model = nn.Sequential(
    nn.Linear(10, 100),
    nn.ReLU(),
    nn.Linear(100, 1)
)

print("Model built. Starting single pass...", flush=True)
x = torch.randn(1, 10)
y = model(x)
print("Result:", y.item(), flush=True)
print("EXECUTION_SUCCESSFUL", flush=True)
