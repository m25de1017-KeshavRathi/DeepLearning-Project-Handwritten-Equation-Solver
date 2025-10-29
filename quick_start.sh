#!/bin/bash

# Quick start script for handwritten equation solver

echo "=========================================="
echo "Handwritten Equation Solver - Quick Start"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Configure Kaggle API (if not already configured)
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo ""
    echo "WARNING: Kaggle API credentials not found!"
    echo "Please place your kaggle.json file in ~/.kaggle/"
    echo "You can download it from: https://www.kaggle.com/settings"
    echo ""
    read -p "Press enter to continue (or Ctrl+C to exit)..."
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data models logs

# Download and parse dataset
echo ""
echo "Downloading and parsing CROHME2019 dataset..."
python src/data_loader.py

# Start training
echo ""
echo "=========================================="
echo "Dataset is ready!"
echo "=========================================="
echo ""
echo "To start training, run:"
echo "  python train.py --epochs 100 --batch_size 32 --augment"
echo ""
echo "For more options, run:"
echo "  python train.py --help"
echo ""
echo "=========================================="

