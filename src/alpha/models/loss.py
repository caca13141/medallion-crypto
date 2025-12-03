import torch
import torch.nn as nn

class OrdinalWassersteinLoss(nn.Module):
    """
    Ordinal Wasserstein Loss (Earth Mover's Distance) for classification.
    Assumes classes are ordered: 0 (Long) <-> 1 (Neutral) <-> 2 (Short)
    Wait, usually 0=Neutral, 1=Long, 2=Short in our mapping?
    
    Let's check SignalEngine mapping:
    # Mapping: 0=LONG, 1=NEUTRAL, 2=SHORT (Arbitrary for MVP)
    # Actually in SignalEngine we defined:
    # target = 1 if signal == 1 else 2 (so 0 is implicitly Neutral?)
    
    Let's standardize:
    0: LONG
    1: NEUTRAL
    2: SHORT
    
    Distance Matrix:
    | L N S |
    L 0 1 2
    N 1 0 1
    S 2 1 0
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        # Create distance matrix
        self.M = torch.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                self.M[i, j] = abs(i - j)
                
    def forward(self, preds, targets):
        """
        preds: [batch_size, num_classes] (Logits or Probs)
        targets: [batch_size] (Class Indices)
        """
        # Convert logits to softmax probs if needed
        if preds.min() < 0 or preds.max() > 1:
            probs = torch.softmax(preds, dim=1)
        else:
            probs = preds
            
        batch_size = probs.size(0)
        loss = 0.0
        
        # Move M to device
        M = self.M.to(probs.device)
        
        for i in range(batch_size):
            target_idx = targets[i]
            # Ground truth distribution (one-hot)
            target_dist = torch.zeros(self.num_classes).to(probs.device)
            target_dist[target_idx] = 1.0
            
            # EMD calculation (Simplified for 1D ordinal)
            # CDF method is efficient for 1D
            pred_cdf = torch.cumsum(probs[i], dim=0)
            target_cdf = torch.cumsum(target_dist, dim=0)
            
            loss += torch.sum(torch.abs(pred_cdf - target_cdf))
            
        return loss / batch_size
