import torch
import sys
import os

# Fix Import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.forecasting.topology_forecaster import create_model

def verify():
    print(" VERIFYING DOWNLOADED MODEL...")
    
    model_path = 'models/transformer_best.pth'
    
    if not os.path.exists(model_path):
        print(f" Error: Model file not found at {model_path}")
        print("   Did you run the scp download command?")
        return

    try:
        # Create Model Architecture
        model = create_model()
        
        # Load State Dict
        # map_location='cpu' ensures it loads even if local machine has no CUDA
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        model.eval()
        
        print("\n MODEL LOADED SUCCESSFULLY!")
        print("-" * 30)
        print(f"Architecture: {model.__class__.__name__}")
        print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")
        print(f"File Size:    {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        print("-" * 30)
        print("Ready for inference.")
        
    except Exception as e:
        print(f"\n FAILED TO LOAD MODEL: {e}")

if __name__ == "__main__":
    verify()
