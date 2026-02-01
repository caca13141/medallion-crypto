#!/bin/bash
# Prepare RunPod Training Package
# Includes full topology dataset for fastest deployment

echo "📦 PREPARING RUNPOD PACKAGE"
echo "=============================="

# Create package directory
mkdir -p runpod_package

# Copy topology dataset (856MB)
echo "Copying topology dataset..."
cp src/data/topology_dataset/production_topology.pkl runpod_package/

# Copy training scripts
echo "Copying training code..."
mkdir -p runpod_package/src/training runpod_package/src/forecasting
cp src/training/train_production_transformer.py runpod_package/src/training/
cp src/forecasting/topology_forecaster.py runpod_package/src/forecasting/

# Create archive
echo "Creating package..."
tar -czf runpod_package.tar.gz -C runpod_package .

# Cleanup
rm -rf runpod_package

SIZE=$(du -h runpod_package.tar.gz | cut -f1)

echo ""
echo "✅ RUNPOD PACKAGE READY"
echo "=============================="
echo "File: runpod_package.tar.gz"
echo "Size: $SIZE"
echo ""
echo "Next steps:"
echo "1. Go to https://runpod.io"
echo "2. Deploy RTX 3090 pod"
echo "3. Upload: scp -P PORT runpod_package.tar.gz root@IP:~/"
echo "4. SSH and run: tar -xzf runpod_package.tar.gz && pip install torch tqdm numpy && python src/training/train_production_transformer.py"
echo ""
echo "Cost: ~\$1 for 5 hours of training"
