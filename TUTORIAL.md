# Handwritten Equation Solver - Complete Tutorial

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Dataset Setup](#dataset-setup)
4. [Understanding the Architecture](#understanding-the-architecture)
5. [Training the Model](#training-the-model)
6. [Evaluation](#evaluation)
7. [Making Predictions](#making-predictions)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

---

## Introduction

This project implements a deep learning model for recognizing and solving handwritten mathematical equations. The system uses:

- **CNN Encoder**: Extracts visual features from handwritten strokes
- **Transformer Decoder**: Generates LaTeX sequences
- **Attention Mechanism**: Aligns visual features with output symbols
- **CROHME2019 Dataset**: Competition-quality handwritten math expressions

### Key Features
- End-to-end trainable architecture
- Support for complex mathematical expressions
- Data augmentation for improved generalization
- Beam search for better predictions
- Comprehensive evaluation metrics

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster training

### Step 1: Clone or Navigate to Project Directory

```bash
cd /path/to/DeepLearning-Project-Handwritten-Equation-Solver
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Kaggle API

The dataset is hosted on Kaggle. You need to set up Kaggle API credentials:

1. Go to https://www.kaggle.com/settings
2. Click "Create New API Token"
3. Save the downloaded `kaggle.json` file to `~/.kaggle/` directory

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Dataset Setup

### Automatic Download and Parsing

The dataset will be automatically downloaded and parsed when you first run the training script. However, you can pre-download it:

```bash
python src/data_loader.py
```

This will:
1. Download CROHME2019 dataset from Kaggle
2. Parse all InkML files
3. Split into train/validation/test sets
4. Cache parsed data for faster future loading

### Dataset Structure

The CROHME2019 dataset contains:
- **InkML files**: XML-based format with stroke coordinates
- **Labels**: Ground truth in LaTeX format
- **Splits**: Pre-defined or auto-generated train/val/test splits

### Visualize Dataset

To explore the dataset visually:

```bash
python visualize_data.py --split train --num_samples 5
```

---

## Understanding the Architecture

### CNN Encoder

The encoder extracts visual features from handwritten images:

```
Input (128×128×1) 
  → Conv2D(64) + MaxPool → Conv2D(128) + MaxPool 
  → Conv2D(256) + MaxPool → Conv2D(512) + MaxPool 
  → Conv2D(512) 
  → Reshape to sequence
```

**Output**: Sequence of 512-dimensional feature vectors

### Transformer Decoder

The decoder generates LaTeX sequences using:
- **Self-Attention**: Captures dependencies in output sequence
- **Cross-Attention**: Attends to encoder features
- **Feed-Forward Networks**: Non-linear transformations

**Key Components**:
- 4 decoder layers (configurable)
- 8 attention heads
- Positional encoding for sequence order

### Training Objective

The model is trained to maximize:

```
P(y₁, y₂, ..., yₙ | x) = ∏ P(yᵢ | y₁, ..., yᵢ₋₁, x)
```

where:
- `x` = input image
- `y₁, ..., yₙ` = output LaTeX sequence

---

## Training the Model

### Basic Training

Start training with default parameters:

```bash
python train.py --epochs 100 --batch_size 32
```

### Recommended Settings

For best results:

```bash
python train.py \
  --epochs 100 \
  --batch_size 32 \
  --d_model 512 \
  --num_heads 8 \
  --num_layers 4 \
  --dff 2048 \
  --dropout 0.1 \
  --augment \
  --img_size 128
```

### Training Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--epochs` | Number of training epochs | 100 |
| `--batch_size` | Batch size | 32 |
| `--d_model` | Model dimension | 512 |
| `--num_heads` | Attention heads | 8 |
| `--num_layers` | Decoder layers | 4 |
| `--dff` | Feed-forward dimension | 2048 |
| `--dropout` | Dropout rate | 0.1 |
| `--augment` | Use data augmentation | False |
| `--img_size` | Image size | 128 |
| `--learning_rate` | Learning rate (None = custom schedule) | None |

### Training with GPU

The script automatically detects and uses available GPUs. To control GPU usage:

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 100 --batch_size 32

# Use CPU only
CUDA_VISIBLE_DEVICES=-1 python train.py --epochs 100 --batch_size 32
```

### Resume Training

To resume from a checkpoint:

```bash
python train.py \
  --resume models/model_run_20241029-120000_epoch_50_val_loss_1.2345.h5 \
  --epochs 100
```

### Monitor Training

Training logs are saved to:
- **TensorBoard**: `logs/run_<timestamp>/`
- **CSV logs**: `logs/training_<timestamp>.csv`

View with TensorBoard:

```bash
tensorboard --logdir logs/
```

---

## Evaluation

### Evaluate on Test Set

```bash
python evaluate.py \
  --model_path models/best_model.h5 \
  --split test \
  --batch_size 32
```

### Evaluate on Validation Set

```bash
python evaluate.py \
  --model_path models/best_model.h5 \
  --split val
```

### Save Predictions

To save predictions for analysis:

```bash
python evaluate.py \
  --model_path models/best_model.h5 \
  --split test \
  --save_predictions predictions.json
```

### Evaluation Metrics

The script reports:
- **Exact Match Accuracy**: Percentage of perfectly predicted sequences
- **Character Accuracy**: Character-level accuracy
- **Loss**: Model loss on the dataset

---

## Making Predictions

### Predict from InkML File

```bash
python predict.py \
  --input path/to/equation.inkml \
  --model_path models/best_model.h5 \
  --visualize
```

### Predict from Image

```bash
python predict.py \
  --input path/to/equation.png \
  --model_path models/best_model.h5 \
  --visualize
```

### Save Prediction

```bash
python predict.py \
  --input path/to/equation.inkml \
  --model_path models/best_model.h5 \
  --output prediction.json
```

---

## Advanced Usage

### Custom Model Architecture

Modify model architecture in `src/model.py` or pass parameters:

```bash
python train.py \
  --d_model 256 \
  --num_heads 4 \
  --num_layers 6 \
  --dff 1024
```

### Data Augmentation

The preprocessor applies:
- Random rotation (-15° to +15°)
- Random scaling (0.9 to 1.1)
- Random translation (-0.1 to +0.1)

Enable with `--augment` flag.

### Custom Learning Rate

```bash
# Fixed learning rate
python train.py --learning_rate 0.001

# Custom schedule with warmup
python train.py --warmup_steps 2000
```

### Hyperparameter Tuning

Key hyperparameters to tune:
1. **d_model**: Model capacity (256, 512, 1024)
2. **num_layers**: Decoder depth (2, 4, 6)
3. **num_heads**: Attention heads (4, 8, 16)
4. **dropout**: Regularization (0.0, 0.1, 0.2)
5. **batch_size**: Training stability (16, 32, 64)

---

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size or model size

```bash
python train.py --batch_size 16 --d_model 256
```

### Issue: Slow Training

**Solution**: Use GPU or reduce model complexity

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Issue: Poor Accuracy

**Solutions**:
1. Train longer: `--epochs 200`
2. Use data augmentation: `--augment`
3. Increase model capacity: `--d_model 512 --num_layers 6`
4. Check dataset quality: `python visualize_data.py`

### Issue: Kaggle API Error

**Solution**: Verify Kaggle credentials

```bash
# Test Kaggle API
kaggle datasets list

# Re-configure if needed
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Issue: Import Errors

**Solution**: Reinstall dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

---

## Example Workflow

Here's a complete workflow from start to finish:

```bash
# 1. Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Download and explore data
python visualize_data.py --num_samples 5

# 3. Quick test (small subset)
python example_usage.py

# 4. Full training
python train.py \
  --epochs 100 \
  --batch_size 32 \
  --augment \
  --d_model 512

# 5. Evaluate
python evaluate.py \
  --model_path models/best_model.h5 \
  --split test \
  --save_predictions results.json

# 6. Make predictions
python predict.py \
  --input test_equation.inkml \
  --model_path models/best_model.h5 \
  --visualize
```

---

## Performance Tips

### Training Speed
- Use GPU with CUDA support
- Increase batch size (if memory allows)
- Use mixed precision training (modify model.py)

### Model Quality
- Train for at least 50-100 epochs
- Use data augmentation
- Monitor validation loss for early stopping
- Try ensemble of multiple models

### Memory Usage
- Reduce batch size
- Reduce model dimension (d_model)
- Use gradient accumulation (modify train.py)

---

## Citation

If you use this project, please cite:

```bibtex
@inproceedings{crohme2019,
  title={CROHME 2019: Competition on Recognition of Handwritten Mathematical Expressions},
  author={Mahdavi, Mahshad and Zanibbi, Richard and Mouchere, Harold},
  booktitle={International Conference on Document Analysis and Recognition},
  year={2019}
}
```

---

## Support

For issues or questions:
1. Check this tutorial
2. Review the README.md
3. Inspect example_usage.py
4. Check configuration in config_example.json

---

**Happy Training! 🚀**

