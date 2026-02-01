import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    import torchcde
except ImportError:
    torchcde = None

class NeuralCDEPredictor(nn.Module):
    """
    Phase 3: Log-Signature + NeuralCDE.
    Processes 7-dim irregular market stream for edge & vol forecasting.
    """
    def __init__(self, input_channels=7, hidden_channels=32, output_channels=2):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        # 1. Log-Signature Initializer
        # We use a static log-signature of the recent window to initialize the CDE state z0
        # Depth 5 Log-Sig of 7 channels is large, we project it.
        # Approx dim: 7 + 7^2/2 ... actually LogSig is smaller.
        # For speed, we use a linear map from raw window summary to z0
        self.initial_network = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 2. Neural CDE Vector Field
        # f(z) -> dz/dt
        self.func = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.SiLU(),
            nn.Linear(128, hidden_channels * input_channels),
            nn.Tanh()
        )
        
        # 3. Readout Heads
        self.edge_head = nn.Linear(hidden_channels, 1)      # Directional Edge
        self.vol_head = nn.Linear(hidden_channels, 1)       # Volatility Forecast

    def forward(self, coeffs):
        """
        Args:
            coeffs: Cubic spline coefficients from torchcde.hermite_cubic_coefficients_with_backward_differences
        """
        if torchcde is None:
            return torch.zeros(coeffs.shape[0], 2)

        # Build Continuous Path
        X = torchcde.CubicSpline(coeffs)
        
        # Initial state z0 from first observation
        z0 = self.initial_network(X.evaluate(0.0))
        
        # Integrate CDE: dz_t = f(z_t) dX_t
        # adjoint=False for speed during inference
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.cde_func,
                              t=X.interval,
                              method='rk4',
                              options=dict(step_size=1e-1))
        
        # Final state
        z_final = z_T[:, -1]
        
        # Heads
        edge = torch.tanh(self.edge_head(z_final))
        vol = F.softplus(self.vol_head(z_final))
        
        return torch.cat([edge, vol], dim=1)

    def cde_func(self, t, z):
        # Reshape for matrix multiplication
        # (Batch, Hidden) -> (Batch, Hidden * Input) -> (Batch, Hidden, Input)
        batch_dims = z.shape[:-1]
        return self.func(z).view(*batch_dims, self.hidden_channels, self.input_channels)

    @staticmethod
    def preprocess_stream(times, values):
        """
        Convert irregular stream to CDE coefficients.
        times: (N,)
        values: (N, 7) [price, bid_vol, ask_vol, trade_buy, trade_sell, canc_buy, canc_sell]
        """
        if torchcde is None: return None
        return torchcde.hermite_cubic_coefficients_with_backward_differences(values, times)
