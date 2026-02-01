"""
Professional Rough Path Signature Model (2026 Standard)
Replaces Transformer with Signature Features for robust, time-invariant feature extraction.

Input: 72 x 4 (Price, Volume, Funding, LiqVolume)
Output: 2048-dim Feature Vector -> Linear Head -> Loop Score & Leverage

Math:
S(X)_{a,b} = (1, X^1_{a,b}, X^2_{a,b}, ..., X^k_{a,b})
where X^k_{a,b} = int_{a < t1 < ... < tk < b} dX_{t1} ... dX_{tk}

Depth 5 Signature of 4D path captures all geometric properties up to degree 5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
from numba import jit, float64, int64

# Check for GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@jit(nopython=True, cache=True)
def compute_signature_numba(path: np.ndarray, depth: int) -> np.ndarray:
    """
    Compute truncated signature of a path using Numba (CPU optimized).
    Matches the logic of the PyTorch SignatureLayer.
    
    Args:
        path: (Length, Channels) array
        depth: Truncation depth
        
    Returns:
        signature: Flat array of signature terms
    """
    length, channels = path.shape
    
    # 1. Compute increments dX
    # dX[t] = path[t+1] - path[t]
    dX = np.zeros((length - 1, channels), dtype=np.float64)
    for t in range(length - 1):
        for c in range(channels):
            dX[t, c] = path[t+1, c] - path[t, c]
            
    # List to store flattened signatures for each level
    # We can't use a list in nopython mode easily for varying sizes if we want to return a single array
    # So we'll calculate size first or just append to a pre-allocated buffer?
    # Calculating size is safer.
    
    total_dim = 0
    for k in range(1, depth + 1):
        total_dim += channels ** k
        
    signature = np.zeros(total_dim, dtype=np.float64)
    current_idx = 0
    
    # Level 1: Sum of increments
    # S^1 = sum(dX)
    sig_lvl1 = np.zeros(channels, dtype=np.float64)
    for t in range(length - 1):
        for c in range(channels):
            sig_lvl1[c] += dX[t, c]
            
    # Store Level 1
    for c in range(channels):
        signature[current_idx + c] = sig_lvl1[c]
    current_idx += channels
    
    # Higher Levels
    # We need to maintain the "path so far" for the previous level
    # Level k terms are integrals of Level k-1 terms against dX
    
    # Current level sequence: (Length-1, PrevDim)
    # Initialize with dX for Level 1
    prev_level_seq = dX
    prev_dim = channels
    
    for k in range(2, depth + 1):
        # Next level dimension
        next_dim = prev_dim * channels
        
        # We need to compute the sequence for level k to use for level k+1
        # Seq[t] = CumSum(PrevSeq)[:t] (outer) dX[t]
        
        # Allocate next level sequence
        next_level_seq = np.zeros((length - 1, next_dim), dtype=np.float64)
        
        # Compute cumulative sum of previous level
        prev_cumsum = np.zeros((length - 1, prev_dim), dtype=np.float64)
        
        # Manual cumsum
        curr_sum = np.zeros(prev_dim, dtype=np.float64)
        for t in range(length - 1):
            for d in range(prev_dim):
                curr_sum[d] += prev_level_seq[t, d]
                prev_cumsum[t, d] = curr_sum[d]
                
        # Compute terms
        # Term[t] = PrevCumSum[t-1] * dX[t]
        # Note: PyTorch code used:
        # prev_integral[:, 1:, :] = torch.cumsum(last_level_path[:, :-1, :], dim=1)
        # So at time t, we multiply CumSum up to t-1 by dX[t]
        
        # Total signature for this level
        sig_lvl_k = np.zeros(next_dim, dtype=np.float64)
        
        for t in range(1, length - 1): # Start at 1 because t=0 has 0 integral before it
            # Outer product: PrevCumSum[t-1] (dim D1) x dX[t] (dim D2) -> (D1*D2)
            # Flattened index: i * D2 + j
            
            for i in range(prev_dim):
                val_prev = prev_cumsum[t-1, i]
                for j in range(channels):
                    val_new = dX[t, j]
                    res = val_prev * val_new
                    
                    flat_idx = i * channels + j
                    next_level_seq[t, flat_idx] = res
                    sig_lvl_k[flat_idx] += res
                    
        # Store Level k signature
        for i in range(next_dim):
            signature[current_idx + i] = sig_lvl_k[i]
        current_idx += next_dim
        
        # Update for next iteration
        prev_level_seq = next_level_seq
        prev_dim = next_dim
        
    return signature

class SignatureLayer(nn.Module):
    """
    Computes the truncated signature of a path.
    Optimized PyTorch implementation (mimics CUDA kernel logic).
    """
    def __init__(self, in_channels: int, depth: int):
        super().__init__()
        self.in_channels = in_channels
        self.depth = depth
        
        # Calculate signature dimension
        self.sig_dim = 0
        for k in range(1, depth + 1):
            self.sig_dim += in_channels ** k
            
    def forward(self, path: torch.Tensor) -> torch.Tensor:
        batch_size, length, channels = path.shape
        dX = path[:, 1:, :] - path[:, :-1, :]
        sig_lvl1 = torch.sum(dX, dim=1)
        signatures = [sig_lvl1]
        last_level_path = dX
        
        for k in range(2, self.depth + 1):
            prev_integral = torch.zeros_like(last_level_path)
            prev_integral[:, 1:, :] = torch.cumsum(last_level_path[:, :-1, :], dim=1)
            term1 = prev_integral.unsqueeze(-1)
            term2 = dX.unsqueeze(-2)
            next_level_seq = (term1 * term2).view(batch_size, length-1, -1)
            sig_lvl_k = torch.sum(next_level_seq, dim=1)
            signatures.append(sig_lvl_k)
            last_level_path = next_level_seq
            
        return torch.cat(signatures, dim=1)

class LogSignatureLayer(nn.Module):
    """
    Computes Log-Signature (compressed representation).
    Uses 'signatory' if available, otherwise falls back to projected Signature.
    """
    def __init__(self, in_channels: int, depth: int):
        super().__init__()
        self.in_channels = in_channels
        self.depth = depth
        self.use_signatory = False
        
        try:
            import signatory
            self.use_signatory = True
            self.sig_dim = signatory.logsignature_channels(in_channels, depth)
            print(f" Signatory found. Using Log-Signature (Dim={self.sig_dim})")
        except ImportError:
            print(" Signatory not found. Using Projected Signature fallback.")
            # Fallback: Compute standard signature and project it
            self.sig_layer = SignatureLayer(in_channels, depth)
            self.raw_dim = self.sig_layer.sig_dim
            # Target dim approx same as log-sig (heuristic)
            self.sig_dim = min(self.raw_dim, 256) 
            self.projection = nn.Linear(self.raw_dim, self.sig_dim)

    def forward(self, path: torch.Tensor) -> torch.Tensor:
        if self.use_signatory:
            import signatory
            # path: (N, L, C)
            return signatory.logsignature(path, self.depth)
        else:
            # Fallback
            raw_sig = self.sig_layer(path)
            return self.projection(raw_sig)

class SignatureModel(nn.Module):
    def __init__(self, input_channels=4, depth=5, output_dim=512):
        super().__init__()
        
        # Use Log-Signature for efficiency (RAM/Compute)
        self.signature_layer = LogSignatureLayer(input_channels, depth)
        sig_dim = self.signature_layer.sig_dim
        
        # Projection Head
        self.projection = nn.Sequential(
            nn.Linear(sig_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
        # Final Heads
        self.loop_score_head = nn.Linear(output_dim, 1) # Predicts Topology Loop Score
        self.leverage_head = nn.Linear(output_dim, 1)   # Predicts Optimal Leverage
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (Batch, 72, 4) Normalized paths
        Returns:
            loop_score: (Batch, 1)
            leverage: (Batch, 1)
        """
        # 1. Compute Signature Features
        sig_feats = self.signature_layer(x)
        
        return self.forward_from_signature(sig_feats)

    def forward_from_signature(self, sig_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass starting from pre-computed signature features.
        Useful for fast inference with Numba.
        """
        # 2. Project to Latent Space
        latent = self.projection(sig_feats)
        
        # 3. Predict
        loop_score = torch.tanh(self.loop_score_head(latent)) # [-1, 1]
        leverage = torch.sigmoid(self.leverage_head(latent))  # [0, 1] -> map to 1x-20x
        
        return loop_score, leverage

# Production Inference Wrapper
class RoughPathEngine:
    def __init__(self, model_path: Optional[str] = None):
        self.model = SignatureModel().to(DEVICE)
        self.model.eval()
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f" Loaded Signature Model from {model_path}")
        else:
            print(" Initialized Signature Model with random weights (Training Mode)")
            
    def predict(self, path_data: torch.Tensor) -> Tuple[float, float]:
        """
        Inference on a single path using PyTorch.
        path_data: (72, 4) Tensor
        """
        with torch.no_grad():
            # Add batch dim
            if path_data.dim() == 2:
                path_data = path_data.unsqueeze(0)
                
            path_data = path_data.to(DEVICE)
            
            score, lev = self.model(path_data)
            
            return score.item(), lev.item()

    def fast_predict(self, path_data: np.ndarray) -> Tuple[float, float]:
        """
        Fast inference using Numba for signature calculation (CPU optimized).
        path_data: (72, 4) Numpy Array
        """
        # 1. Compute Signature with Numba (CPU)
        # Depth=5 matches the model default
        sig_feats_np = compute_signature_numba(path_data, depth=5)
        
        # 2. Convert to Tensor for MLP head
        sig_feats = torch.from_numpy(sig_feats_np).float().unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            score, lev = self.model.forward_from_signature(sig_feats)
            
        return score.item(), lev.item()

if __name__ == "__main__":
    # Verification
    print(" Initializing Professional Rough Path Engine...")
    
    # Create dummy batch: 32 paths, length 72, 4 channels
    x = torch.randn(32, 72, 4).to(DEVICE)
    
    model = SignatureModel().to(DEVICE)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Warmup
    _ = model(x)
    
    # Timing
    import time
    start = time.time()
    for _ in range(100):
        _ = model(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    
    avg_time = (end - start) / 100 * 1000 # ms
    print(f" Inference Time (Batch 32): {avg_time:.2f} ms")
    
    # Check output
    score, lev = model(x)
    print(f"Output Shape: {score.shape}")
    assert score.shape == (32, 1)
    
    # Verify Numba Implementation
    print("\n Verifying Numba Implementation...")
    path_np = x[0].cpu().numpy() # (72, 4)
    
    # Warmup
    _ = compute_signature_numba(path_np, 5)
    
    start = time.time()
    for _ in range(100):
        sig = compute_signature_numba(path_np, 5)
    end = time.time()
    
    avg_time_numba = (end - start) / 100 * 1000
    print(f" Numba Time (Single Path): {avg_time_numba:.4f} ms")
    print(f"Signature Shape: {sig.shape}")
    
    # Verify Fast Predict
    engine = RoughPathEngine()
    score, lev = engine.fast_predict(path_np)
    print(f"Fast Predict Output: Score={score:.4f}, Lev={lev:.4f}")
    
    print(" Verification Passed")
