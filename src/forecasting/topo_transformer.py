"""
Topological Transformer: 72h Persistence Images → 48h Loop Dissolution Forecast
Input: Sequence of persistence images (not raw price)
Output: Time + strength of next H1 loop dissolution (>92% accuracy target)
"""
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

class TopologicalTransformer(nn.Module):
    """
    Transformer that operates on persistence image sequences.
    
    Architecture:
    - Input: (batch, seq_len=72, img_h=20, img_w=20) persistence images
    - Flatten each image to tokens
    - Multi-head self-attention across time
    - Output: (time_to_dissolution, dissolution_strength, confidence)
    """
    
    def __init__(
        self,
        img_size=20,
        seq_len=72,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    ):
        super().__init__()
        
        self.img_size = img_size
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Persistence image → embedding
        self.img_embed = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 20→10
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 10→5
            nn.ReLU(),
            nn.Flatten(),  # 64*5*5 = 1600
            nn.Linear(1600, d_model)
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Prediction heads
        self.time_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 48)  # 48 time bins (1h each for 48h ahead)
        )
        
        self.strength_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 0-1 normalized strength
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # prediction confidence
        )
        
    def forward(self, persistence_images):
        """
        Args:
            persistence_images: (batch, seq_len, 1, img_h, img_w)
            
        Returns:
            time_logits: (batch, 48) - probability distribution over time bins
            strength: (batch, 1) - dissolution strength [0, 1]
            confidence: (batch, 1) - prediction confidence [0, 1]
        """
        batch, seq_len, _, h, w = persistence_images.shape
        
        # Flatten sequence for conv
        imgs = rearrange(persistence_images, 'b s c h w -> (b s) c h w')
        
        # Embed each image
        embeddings = self.img_embed(imgs)  # (b*s, d_model)
        embeddings = rearrange(embeddings, '(b s) d -> b s d', b=batch, s=seq_len)
        
        # Add positional encoding
        embeddings = embeddings + self.pos_encoding[:, :seq_len, :]
        
        # Transformer
        features = self.transformer(embeddings)  # (batch, seq_len, d_model)
        
        # Use last token for prediction (like BERT [CLS])
        final_repr = features[:, -1, :]
        
        # Predict
        time_logits = self.time_head(final_repr)  # (batch, 48)
        strength = self.strength_head(final_repr)  # (batch, 1)
        confidence = self.confidence_head(final_repr)  # (batch, 1)
        
        return time_logits, strength, confidence
    
    def predict(self, persistence_images):
        """
        Inference mode: return most likely time, strength, and confidence.
        
        Returns:
            time_hours: predicted hours until next H1 dissolution
            strength: dissolution strength [0, 1]
            confidence: prediction confidence [0, 1]
        """
        self.eval()
        with torch.no_grad():
            time_logits, strength, confidence = self.forward(persistence_images)
            
            # Get most likely time bin
            time_probs = torch.softmax(time_logits, dim=-1)
            time_bin = torch.argmax(time_probs, dim=-1)
            
            return time_bin.cpu().numpy(), strength.cpu().numpy(), confidence.cpu().numpy()


class TopoTransformerTrainer:
    """
    Training loop for TopologicalTransformer.
    Loss = CrossEntropy(time) + MSE(strength) + ConfidencePenalty
    """
    
    def __init__(self, model, lr=1e-4, weight_decay=1e-5):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
    def train_step(self, persistence_seq, target_time, target_strength):
        """
        Single training step.
        
        Args:
            persistence_seq: (batch, 72, 1, 20, 20) - 72h of persistence images
            target_time: (batch,) - ground truth time bin [0, 47]
            target_strength: (batch, 1) - ground truth strength [0, 1]
            
        Returns:
            loss: scalar loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward
        time_logits, pred_strength, confidence = self.model(persistence_seq)
        
        # Losses
        loss_time = self.ce_loss(time_logits, target_time)
        loss_strength = self.mse_loss(pred_strength, target_strength)
        
        # Confidence penalty: penalize overconfident wrong predictions
        time_probs = torch.softmax(time_logits, dim=-1)
        predicted_time = torch.argmax(time_probs, dim=-1)
        is_correct = (predicted_time == target_time).float().unsqueeze(1)
        
        # If wrong and confident, penalty
        confidence_penalty = torch.mean((1 - is_correct) * confidence)
        
        # Total loss
        total_loss = loss_time + 0.5 * loss_strength + 0.1 * confidence_penalty
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item()
    
    def evaluate(self, persistence_seq, target_time, target_strength):
        """
        Evaluate accuracy on validation set.
        
        Returns:
            accuracy: time prediction accuracy
            mae_strength: MAE for strength prediction
        """
        self.model.eval()
        with torch.no_grad():
            time_logits, pred_strength, confidence = self.model(persistence_seq)
            
            # Time accuracy
            time_probs = torch.softmax(time_logits, dim=-1)
            predicted_time = torch.argmax(time_probs, dim=-1)
            accuracy = (predicted_time == target_time).float().mean().item()
            
            # Strength MAE
            mae_strength = torch.abs(pred_strength - target_strength).mean().item()
            
            return accuracy, mae_strength
