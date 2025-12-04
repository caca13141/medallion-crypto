"""
MAXIMUM LEARNING PIPELINE
Trains until complete convergence - no holds barred.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import os
from datetime import datetime

class MaxLearningPipeline:
    """
    Aggressive training until convergence.
    No early stopping until TRUE plateau.
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 50  # VERY patient (normal is 10)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epochs': []
        }
        
    def train_until_dead(self, train_loader, val_loader, max_epochs=1000):
        """
        Train until the model literally cannot improve anymore.
        """
        
        # Optimizer with learning rate scheduling
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Aggressive LR schedule: Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2
        )
        
        # Loss function
        criterion = nn.MSELoss()
        
        print("🔥 MAXIMUM LEARNING MODE ACTIVATED")
        print(f"Device: {self.device}")
        print(f"Model: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M params")
        print(f"Max Epochs: {max_epochs}")
        print(f"Patience: {self.max_patience} (aggressive)")
        print("=" * 60)
        
        for epoch in range(max_epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for batch_idx, (X, y_scalars, y_vectors, y_images) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                X = X.to(self.device)
                y_scalars = y_scalars.to(self.device)
                y_vectors = y_vectors.to(self.device)
                y_images = y_images.to(self.device)
                
                # Forward
                pred_scalars, pred_vectors, pred_images = self.model(X)
                
                # Multi-task loss
                loss_scalar = criterion(pred_scalars, y_scalars)
                loss_vector = criterion(pred_vectors, y_vectors)
                loss_image = criterion(pred_images, y_images)
                
                # Weighted combination
                loss = loss_scalar + 0.5 * loss_vector + 0.3 * loss_image
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping (prevent explosion)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation phase
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for X, y_scalars, y_vectors, y_images in val_loader:
                    X = X.to(self.device)
                    y_scalars = y_scalars.to(self.device)
                    y_vectors = y_vectors.to(self.device)
                    y_images = y_images.to(self.device)
                    
                    pred_scalars, pred_vectors, pred_images = self.model(X)
                    
                    loss = (criterion(pred_scalars, y_scalars) + 
                           0.5 * criterion(pred_vectors, y_vectors) +
                           0.3 * criterion(pred_images, y_images))
                    
                    val_losses.append(loss.item())
            
            # Metrics
            avg_train = np.mean(train_losses)
            avg_val = np.mean(val_losses)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Logging
            self.history['train_loss'].append(avg_train)
            self.history['val_loss'].append(avg_val)
            self.history['learning_rates'].append(current_lr)
            self.history['epochs'].append(epoch + 1)
            
            print(f"\nEpoch {epoch+1}/{max_epochs}")
            print(f"  Train Loss: {avg_train:.6f}")
            print(f"  Val Loss: {avg_val:.6f}")
            print(f"  LR: {current_lr:.2e}")
            
            # Save best model
            if avg_val < self.best_loss:
                improvement = self.best_loss - avg_val
                self.best_loss = avg_val
                self.patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': self.best_loss,
                    'history': self.history
                }, 'models/topolomega_best.pth')
                
                print(f"  ✅ NEW BEST! (improved by {improvement:.6f})")
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{self.max_patience})")
            
            # Learning rate scheduling
            scheduler.step()
            
            # Aggressive early stopping (only after VERY long plateau)
            if self.patience_counter >= self.max_patience:
                print(f"\n🛑 CONVERGENCE REACHED after {epoch+1} epochs")
                print(f"Best Val Loss: {self.best_loss:.6f}")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, optimizer)
        
        # Save final training history
        self.save_history()
        
        print("\n" + "="*60)
        print("🏁 TRAINING COMPLETE")
        print("="*60)
        print(f"Total Epochs: {epoch+1}")
        print(f"Best Val Loss: {self.best_loss:.6f}")
        print(f"Model saved: models/topolomega_best.pth")
        
        return self.history
    
    def save_checkpoint(self, epoch, optimizer):
        """Save periodic checkpoint"""
        os.makedirs('models/checkpoints', exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': self.history
        }, f'models/checkpoints/checkpoint_epoch_{epoch+1}.pth')
    
    def save_history(self):
        """Save training history to JSON"""
        os.makedirs('results', exist_ok=True)
        with open('results/training_history.json', 'w') as f:
            # Convert numpy arrays to lists for JSON
            history_json = {k: [float(x) for x in v] for k, v in self.history.items()}
            json.dump(history_json, f, indent=2)

if __name__ == "__main__":
    print("⚠️  Run this via train_full_pipeline.py")
    print("This is a module, not a standalone script.")
