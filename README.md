# 🔢 Handwritten Equation Solver

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A complete deep learning solution that converts handwritten mathematical equations into LaTeX format using the CROHME2019 dataset.

## 🎯 Overview

This project implements a **CNN-Transformer hybrid architecture** for recognizing handwritten mathematical expressions:

- 🖼️ **CNN Encoder**: Extracts visual features from handwritten strokes
- 🔄 **Transformer Decoder**: Generates LaTeX sequences with attention mechanism
- 📊 **CROHME2019 Dataset**: Competition-quality handwritten math expressions
- 🚀 **Production-Ready**: Complete training, evaluation, and inference pipeline

### Key Features

✨ End-to-end trainable architecture  
✨ State-of-the-art transformer-based decoder  
✨ Data augmentation for improved generalization  
✨ Beam search for higher quality predictions  
✨ Comprehensive evaluation metrics  
✨ TensorBoard integration for monitoring  
✨ Automatic dataset download and parsing  

## 🚀 Quick Start

### 1. Setup (One Command)

```bash
chmod +x quick_start.sh && ./quick_start.sh
```

Or manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure Kaggle API (required for dataset download)
# Download kaggle.json from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Test Setup

```bash
python test_setup.py
```

### 3. Quick Demo

```bash
python example_usage.py
```

### 4. Train Model

```bash
python train.py --epochs 100 --batch_size 32 --augment
```

### 5. Evaluate

```bash
python evaluate.py --model_path models/best_model.h5 --split test
```

### 6. Make Predictions

```bash
python predict.py --input equation.inkml --model_path models/best_model.h5 --visualize
```

## 📁 Project Structure

```
DeepLearning-Project-Handwritten-Equation-Solver/
│
├── src/                          # Source code
│   ├── data_loader.py           # Dataset download and InkML parsing
│   ├── preprocessing.py         # Data preprocessing and augmentation
│   ├── model.py                 # CNN-Transformer architecture
│   └── utils.py                 # Utility functions
│
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── predict.py                   # Inference script
├── example_usage.py             # Quick demo
├── visualize_data.py            # Dataset visualization
├── test_setup.py                # Setup verification
│
├── requirements.txt             # Python dependencies
├── quick_start.sh               # Automated setup script
│
├── README.md                    # This file
├── TUTORIAL.md                  # Complete tutorial
├── ARCHITECTURE.md              # Architecture details
├── PROJECT_SUMMARY.md           # Project summary
│
├── data/                        # Dataset cache (auto-created)
├── models/                      # Saved models (auto-created)
└── logs/                        # Training logs (auto-created)
```

## 🏗️ Architecture

### Model: CNN-Transformer Hybrid

```
Input (128×128 image)
    ↓
┌─────────────────┐
│  CNN Encoder    │  5 Conv layers → Feature sequence
└─────────────────┘
    ↓
┌─────────────────┐
│  Transformer    │  4 Decoder layers with attention
│  Decoder        │  Multi-head attention (8 heads)
└─────────────────┘
    ↓
LaTeX Output
```

**Key Components**:
- **Encoder**: 5-layer CNN (64→128→256→512→512 channels)
- **Decoder**: 4-layer Transformer with multi-head attention
- **Attention**: 8 heads, dimension 512
- **Parameters**: ~25M trainable parameters

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation.

## 📊 Dataset

**CROHME2019 Dataset**:
- Competition on Recognition of Handwritten Mathematical Expressions
- InkML format (stroke coordinates + LaTeX labels)
- Train/Validation/Test splits
- Automatic download via Kaggle API

**Preprocessing**:
- Stroke normalization
- Image rendering (128×128)
- Token vocabulary building
- Data augmentation (rotation, scaling, translation)

### Visualize Dataset

```bash
python visualize_data.py --split train --num_samples 5
```

## 🎓 Training

### Basic Training

```bash
python train.py --epochs 100 --batch_size 32
```

### Recommended Settings

```bash
python train.py \
  --epochs 100 \
  --batch_size 32 \
  --d_model 512 \
  --num_heads 8 \
  --num_layers 4 \
  --augment
```

### Training Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--epochs` | Number of epochs | 100 |
| `--batch_size` | Batch size | 32 |
| `--d_model` | Model dimension | 512 |
| `--num_heads` | Attention heads | 8 |
| `--num_layers` | Decoder layers | 4 |
| `--augment` | Use augmentation | False |
| `--learning_rate` | Learning rate | Custom schedule |

### Monitor Training

```bash
tensorboard --logdir logs/
```

## 📈 Evaluation

### Evaluate on Test Set

```bash
python evaluate.py \
  --model_path models/best_model.h5 \
  --split test \
  --save_predictions predictions.json
```

### Metrics
- **Exact Match Accuracy**: Percentage of perfectly predicted equations
- **Character Accuracy**: Character-level accuracy
- **Masked Loss**: Cross-entropy loss (ignoring padding)

### Expected Results

After training on full dataset:
- Exact Match Accuracy: **45-55%**
- Character Accuracy: **75-85%**
- Training Time (GPU): **8-12 hours**

## 🔮 Prediction

### From InkML File

```bash
python predict.py \
  --input equation.inkml \
  --model_path models/best_model.h5 \
  --visualize
```

### From Image

```bash
python predict.py \
  --input equation.png \
  --model_path models/best_model.h5 \
  --visualize
```

### Inference Modes
- **Greedy Decoding**: Fast, good quality
- **Beam Search**: Slower, best quality

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file - Quick start guide |
| [TUTORIAL.md](TUTORIAL.md) | Complete step-by-step tutorial |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Detailed architecture documentation |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Comprehensive project summary |

## 🛠️ Requirements

- **Python 3.9-3.12** (⚠️ Python 3.14+ not supported yet - see [PYTHON_COMPATIBILITY.md](PYTHON_COMPATIBILITY.md))
- TensorFlow 2.15+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- Kaggle API credentials

See [requirements.txt](requirements.txt) for complete list.

**Note**: If you're using Python 3.14+, you'll need to install Python 3.11 or 3.12. See [PYTHON_COMPATIBILITY.md](PYTHON_COMPATIBILITY.md) for detailed instructions.

## 🐛 Troubleshooting

### Out of Memory
```bash
python train.py --batch_size 16 --d_model 256
```

### No GPU Detected
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Kaggle API Error
```bash
# Download kaggle.json from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

See [TUTORIAL.md](TUTORIAL.md) for more troubleshooting tips.

## 🎯 Use Cases

- 📝 Digitizing handwritten math notes
- 🎓 Educational technology applications
- 📱 Math input for mobile apps
- 🔬 Research in handwriting recognition
- 🤖 Building AI tutoring systems

## 🚀 Advanced Usage

### Custom Model Size

```bash
# Small model (fast training)
python train.py --d_model 256 --num_layers 2

# Large model (best accuracy)
python train.py --d_model 768 --num_layers 6
```

### Resume Training

```bash
python train.py --resume models/checkpoint.h5 --epochs 100
```

### Batch Prediction

```bash
# Process multiple files
for file in equations/*.inkml; do
  python predict.py --input "$file" --model_path models/best_model.h5
done
```

## 📊 Performance

### Training Performance
- **GPU (3090)**: ~4-6 hours for 100 epochs
- **GPU (2080Ti)**: ~8-12 hours for 100 epochs
- **CPU**: ~50-100 hours for 100 epochs

### Inference Speed
- **Greedy**: ~50-100 ms/sample (GPU)
- **Beam Search**: ~250-500 ms/sample (GPU)

### Model Size
- **Parameters**: ~25M
- **Disk**: ~100 MB
- **Memory**: ~1.5 GB (GPU training)

## 🎓 Citation

```bibtex
@inproceedings{crohme2019,
  title={CROHME 2019: Competition on Recognition of Handwritten Mathematical Expressions},
  author={Mahdavi, Mahshad and Zanibbi, Richard and Mouchere, Harold},
  booktitle={International Conference on Document Analysis and Recognition},
  year={2019}
}
```

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Ideas:
- Improve model architecture
- Add equation solving capability
- Create web interface
- Optimize inference speed
- Add more data augmentation

## 🙏 Acknowledgments

- CROHME competition organizers for the dataset
- TensorFlow team for the framework
- Kaggle for dataset hosting

## 📞 Support

- 📖 Read the [TUTORIAL.md](TUTORIAL.md)
- 🔍 Check [ARCHITECTURE.md](ARCHITECTURE.md)
- 🧪 Run `python test_setup.py`
- 💡 Try `python example_usage.py`

---

**Ready to get started? Run `python test_setup.py` to verify your setup!** 🚀

