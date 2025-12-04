"""
Transformer Training Script for H1 Dissolution Forecasting
Trains on full persistence image sequences once dataset is ready
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import pickle
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.forecasting.topo_transformer import TopologicalTransformer, TopoTransformerTrainer

class PersistenceImageDataset(Dataset):
    """Dataset for persistence image sequences"""
    
    def __init__(self, data_dict):
        self.images = torch.from_numpy(np.array(data_dict['persistence_images_h1'])).float()
        self.labels_time = torch.from_numpy(data_dict['labels_time']).long()
        self.labels_strength = torch.from_numpy(data_dict['labels_strength']).float()
        
    def __len__(self):
        return len(self.labels_time)
    
    def __getitem__(self, idx):
        # Return single image for now, will need to create sequences
        img = self.images[idx].unsqueeze(0).unsqueeze(0)  # (1, 1, 20, 20)
        return img, self.labels_time[idx], self.labels_strength[idx].unsqueeze(0)

def create_sequences(images, seq_len=72):
    """Create sliding window sequences of persistence images"""
    sequences = []
    labels_time = []
    labels_strength = []
    
    for i in range(len(images) - seq_len):
        seq = images[i:i+seq_len]  # (72, 20, 20)
        sequences.append(seq)
        labels_time.append(images.labels_time[i+seq_len])
        labels_strength.append(images.labels_strength[i+seq_len])
    
    return sequences, labels_time, labels_strength

def train_transformer():
    """Main training loop"""
    
    print("Loading topology dataset...")
    
    # Check if full dataset exists
    if not os.path.exists('src/data/topology_dataset/train.pkl'):
        print("❌ Full topology dataset not ready yet.")
        print("⏳ Waiting for topology_dataset_generator.py to complete...")
        print("\nRun this script again once dataset generation is complete.")
        return
    
    # Load datasets
    with open('src/data/topology_dataset/train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    with open('src/data/topology_dataset/val.pkl', 'rb') as f:
        val_data = pickle.load(f)
    
    print(f"Train samples: {len(train_data['loop_scores'])}")
    print(f"Val samples: {len(val_data['loop_scores'])}")
    
    # For now, use simplified version without sequences
    # Just predicting from single image (can upgrade to 72-image sequences later)
    
    train_dataset = PersistenceImageDataset(train_data)
    val_dataset = PersistenceImageDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create model - simplified for single image input
    print("\nInitializing Transformer...")
    model = TopologicalTransformer(
        img_size=20,
        seq_len=1,  # Single image for now
        d_model=128,
        nhead=4,
        num_layers=3,
        dropout=0.1
    )
    
    trainer = TopoTransformerTrainer(model, lr=1e-4)
    
    # Training loop
    print("\n🚀 Starting training...")
    best_val_acc = 0
    patience = 10
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_acc': [],
        'val_mae_strength': []
    }
    
    for epoch in range(100):
        # Train
        model.train()
        train_losses = []
        
        for batch_idx, (imgs, times, strengths) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            loss = trainer.train_step(imgs, times, strengths)
            train_losses.append(loss)
        
        avg_train_loss = np.mean(train_losses)
        
        # Validate
        val_acc, val_mae = trainer.evaluate(val_loader.dataset.images, 
                                             val_loader.dataset.labels_time,
                                             val_loader.dataset.labels_strength)
        
        history['train_loss'].append(avg_train_loss)
        history['val_acc'].append(val_acc)
        history['val_mae_strength'].append(val_mae)
        
        print(f"Epoch {epoch+1}: Loss={avg_train_loss:.4f}, Val Acc={val_acc:.3f}, Val MAE={val_mae:.3f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'src/data/topo_transformer_trained.pth')
            print(f"✅ New best model saved (acc={val_acc:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Save training history
    with open('results/transformer_training_log.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: src/data/topo_transformer_trained.pth")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    train_transformer()
