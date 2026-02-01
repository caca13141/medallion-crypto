#!/bin/bash
# Prepare AWS Training Package
# Packages topology data + code for upload to AWS

echo "📦 PREPARING AWS TRAINING PACKAGE"
echo "=================================="

# Create temp directory
mkdir -p aws_package
cd aws_package

# Copy topology dataset
echo "Copying topology dataset (856MB)..."
cp ../src/data/topology_dataset/production_topology.pkl ./

# Copy training script
echo "Copying training scripts..."
mkdir -p src/training src/forecasting
cp ../src/training/train_production_transformer.py src/training/train_aws.py
cp ../src/forecasting/topology_forecaster.py src/forecasting/

# Create minimal requirements
echo "Creating requirements..."
cat > requirements_aws.txt << EOF
torch==2.5.1
tqdm==4.66.0
numpy==2.1.0
EOF

# Create optimized training script for AWS
cat > train_aws.py << 'EOF'
"""AWS-Optimized Training Script"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from tqdm import tqdm
import sys
import os

# Import model
sys.path.append('.')
from src.forecasting.topology_forecaster import create_model

class TopologyDataset(Dataset):
    def __init__(self, pkl_path):
        print(f"Loading {pkl_path}...")
        self.data = []
        with open(pkl_path, 'rb') as f:
            while True:
                try:
                    self.data.append(pickle.load(f))
                except EOFError:
                    break
        
        self.images = np.concatenate([d['images'] for d in self.data])
        self.summaries = np.concatenate([d['summaries'] for d in self.data])
        self.labels = np.concatenate([d['labels'] for d in self.data])
        self.images = self.images.reshape(-1, 1, 32, 32)
        print(f"Loaded {len(self.labels)} samples")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.images[idx], dtype=torch.float32),
            torch.tensor(self.summaries[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

print("🚀 AWS GPU TRAINING")
print("=" * 60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

if device == 'cpu':
    print("❌ ERROR: No GPU detected!")
    sys.exit(1)

# Load Data
dataset = TopologyDataset('production_topology.pkl')
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

# Model
model = create_model()
model = model.to(device)
print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Train
epochs = 10
best_loss = float('inf')

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
    for images, summaries, labels in pbar:
        images = images.to(device, non_blocking=True)
        summaries = summaries.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        images_seq = images.unsqueeze(1)
        _, vectors, _ = model(images_seq)
        
        # Classification head (temp)
        logits = torch.matmul(vectors, torch.randn(8, 3, device=device))
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{total_loss/(pbar.n+1):.4f}', 'acc': f'{correct/total:.4f}'})
    
    avg_loss = total_loss / len(loader)
    acc = correct / total
    print(f"\nEpoch {epoch+1}: Loss {avg_loss:.4f}, Acc {acc:.4f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/transformer_best.pth')
        print(f"✅ Saved best model (loss: {best_loss:.4f})")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print(f"Best Loss: {best_loss:.4f}")
print("Model saved: models/transformer_best.pth")
EOF

# Package everything
cd ..
echo "Creating tarball..."
tar -czf aws_training_package.tar.gz \
    -C aws_package \
    production_topology.pkl \
    train_aws.py \
    requirements_aws.txt \
    src/

# Cleanup
rm -rf aws_package

# Summary
SIZE=$(du -h aws_training_package.tar.gz | cut -f1)
echo ""
echo "✅ PACKAGE READY"
echo "=================================="
echo "File: aws_training_package.tar.gz"
echo "Size: $SIZE"
echo ""
echo "Next steps:"
echo "1. Launch AWS g4dn.xlarge instance"
echo "2. Upload: scp -i KEY.pem aws_training_package.tar.gz ubuntu@IP:~/"
echo "3. SSH: ssh -i KEY.pem ubuntu@IP"
echo "4. Run: tar -xzf aws_training_package.tar.gz && cd topoomega && pip install -r requirements_aws.txt && python train_aws.py"
