import torch
import torch.nn as nn
import os
from src.forecasting.topology_forecaster import TopoTransformerGPT

print("Testing Torch Save...")
model = TopoTransformerGPT(d_model=256, nhead=8, num_layers=4)
save_path = os.path.join(os.getcwd(), "test_save.pt")
torch.save(model.state_dict(), save_path)
if os.path.exists(save_path):
    print(f"SUCCESS: Saved to {save_path}")
    os.remove(save_path)
else:
    print("FAILED: File not created")
