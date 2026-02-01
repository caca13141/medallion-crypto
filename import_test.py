import sys
import os
sys.path.append(os.getcwd())

print("Testing Torch...", flush=True)
import torch
print("Torch OK", flush=True)

print("Testing Topology Forecaster...", flush=True)
try:
    from src.forecasting.topology_forecaster import TopoTransformerGPT
    print("Transformer OK", flush=True)
except Exception as e:
    print(f"Transformer FAILED: {e}", flush=True)

print("Testing Intelligence Audit...", flush=True)
try:
    from src.training.intelligence_audit import audit_engine
    print("Audit OK", flush=True)
except Exception as e:
    print(f"Audit FAILED: {e}", flush=True)

print("Testing Dataset Check...", flush=True)
dataset_path = "src/data/topology_dataset/synthetic"
if os.path.exists(dataset_path):
    print(f"Dataset exists at {dataset_path}", flush=True)
    files = os.listdir(dataset_path)
    print(f"Samples found: {len(files)}", flush=True)
else:
    print(f"Dataset NOT FOUND at {dataset_path}", flush=True)

print("IMPORT_TEST_COMPLETE", flush=True)
