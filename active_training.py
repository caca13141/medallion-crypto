import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from src.forecasting.topology_forecaster import TopoTransformerGPT
from src.training.intelligence_audit import audit_engine

# 1. Force CPU for stability during weight initialization
device = "cpu"

print(f"EXASCALE ENGINE: Initializing on {device}...", flush=True)

# 2. Setup Stability Mode Prototype (512 dim, 12 layers)
model = TopoTransformerGPT(d_model=512, nhead=8, num_layers=12, use_checkpointing=False).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print("BOOTSTRAP: Loading Synthetic Dataset...", flush=True)
dataset_path = "src/data/topology_dataset/synthetic"
samples = [f for f in os.listdir(dataset_path) if f.endswith(".pt")]
print(f"FOUND: {len(samples)} market snapshots.", flush=True)

# 3. Training Loop
print("INITIATING CONVERGENCE LOOP. Watch the dashboard 'Intelligence Audit'...", flush=True)

for epoch in range(1):
    total_loss = 0
    for i, sample_name in enumerate(samples[:5]): # 5 steps
        sample_path = os.path.join(dataset_path, sample_name)
        x, label = torch.load(sample_path)
        
        # Prep tensors
        x = x.unsqueeze(0).to(device) # (1, 72, 1, 32, 32)
        target = torch.tensor([label]).float().to(device)
        
        optimizer.zero_grad()
        scalars, vectors, next_img = model(x)
        
        # Loss: TTI Prediction MSE
        loss = F.mse_loss(scalars[:, 1], target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Stream telemetry to dashboard via Intelligence Audit
        # if i % 5 == 0:
        #     # Pseudo-experts for telemetry integration
        #     e_weights = np.random.dirichlet(np.ones(8), size=1)[0]
        #     audit_engine.log_step(loss.item(), e_weights, 0.5 * (1.0/(1.0+loss.item())))
        #     print(f"Epoch {epoch} | Step {i} | Loss: {loss.item():.6f}", flush=True)

    print(f"EPOCH {epoch} COMPLETE. Avg Loss: {total_loss/50:.6f}", flush=True)

# 4. Persist Intuition
save_dir = os.path.join(os.getcwd(), "src", "data", "models")
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, "exascale_bootstrap.pt")
print(f"DEBUG: Attempting to save to {model_save_path}", flush=True)
torch.save(model.cpu().state_dict(), model_save_path) # Save as CPU for portability
print(f"MODEL_SAVED: {model_save_path}", flush=True)

print("EXASCALE_BOOTSTRAP_COMPLETE", flush=True)
