# 🚀 Getting Started Guide

Welcome! This guide will help you get your Handwritten Equation Solver up and running in just a few minutes.

---

## ⚡ Quick Start (5 Minutes)

### Step 1: Verify Python Installation

```bash
python3 --version  # Should be 3.8 or higher
```

If Python is not installed, download it from [python.org](https://www.python.org/downloads/).

### Step 2: Navigate to Project Directory

```bash
cd /Users/chiragphor/Development/projects/DeepLearning-Project-Handwritten-Equation-Solver
```

### Step 3: Run the Quick Start Script

```bash
chmod +x quick_start.sh
./quick_start.sh
```

This will:
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Set up directory structure
- ✅ Check Kaggle API configuration
- ✅ Download and parse the dataset

### Step 4: Test Your Setup

```bash
source venv/bin/activate  # Activate virtual environment
python test_setup.py      # Verify everything works
```

### Step 5: Run Quick Demo

```bash
python example_usage.py
```

**That's it! You're ready to train your first model! 🎉**

---

## 📋 Manual Setup (If Quick Start Fails)

### 1. Create Virtual Environment

```bash
python3 -m venv venv
```

**Activate it:**
- macOS/Linux: `source venv/bin/activate`
- Windows: `venv\Scripts\activate`

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- TensorFlow (deep learning framework)
- OpenCV (image processing)
- NumPy, Pandas (data manipulation)
- Matplotlib (visualization)
- Kaggle Hub (dataset download)
- And more...

### 3. Configure Kaggle API

The CROHME2019 dataset is hosted on Kaggle. You need API credentials:

**a) Get Your Kaggle API Key:**
1. Go to https://www.kaggle.com
2. Sign in (create account if needed)
3. Go to Account Settings: https://www.kaggle.com/settings
4. Scroll to "API" section
5. Click "Create New API Token"
6. Save the downloaded `kaggle.json` file

**b) Install Kaggle Credentials:**

```bash
# Create Kaggle directory
mkdir -p ~/.kaggle

# Move the downloaded file
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set correct permissions (important!)
chmod 600 ~/.kaggle/kaggle.json
```

**c) Verify Configuration:**

```bash
kaggle datasets list
```

If this works, you're all set! If not, check the troubleshooting section below.

### 4. Download and Parse Dataset

```bash
python src/data_loader.py
```

This will:
- Download CROHME2019 dataset (~500MB)
- Parse all InkML files
- Create train/val/test splits
- Cache the processed data

**Time**: 10-30 minutes depending on your internet speed

### 5. Test Your Setup

```bash
python test_setup.py
```

You should see all tests pass with green checkmarks ✓

---

## 🎯 Your First Training Session

### Option 1: Quick Demo (2 Epochs, Small Subset)

```bash
python example_usage.py
```

**What it does:**
- Loads a small subset of data
- Builds a small model
- Trains for 2 epochs
- Shows example prediction

**Time**: 5-10 minutes  
**Purpose**: Verify everything works

### Option 2: Small Training Run (10 Epochs)

```bash
python train.py --epochs 10 --batch_size 16
```

**What it does:**
- Trains on full dataset
- 10 epochs (not enough for good accuracy, but good for testing)
- Smaller batch size (works on most GPUs)

**Time**: 1-2 hours (GPU) or 5-10 hours (CPU)  
**Purpose**: Test full pipeline

### Option 3: Full Training (100 Epochs) - Recommended

```bash
python train.py --epochs 100 --batch_size 32 --augment
```

**What it does:**
- Trains on full dataset with augmentation
- 100 epochs for good accuracy
- Saves best model

**Time**: 8-12 hours (GPU) or 2-3 days (CPU)  
**Purpose**: Get production-quality model

**Pro Tip**: Run in background with nohup:
```bash
nohup python train.py --epochs 100 --batch_size 32 --augment > training.log 2>&1 &
```

---

## 📊 Monitor Training

### Option 1: Watch Console Output

The training script prints progress after each epoch:
```
Epoch 1/100
312/312 [==============================] - 45s 144ms/step - loss: 2.3456 - val_loss: 2.1234
```

### Option 2: TensorBoard (Recommended)

In a new terminal:

```bash
cd /path/to/project
source venv/bin/activate
tensorboard --logdir logs/
```

Then open: http://localhost:6006

You'll see:
- Loss curves
- Accuracy metrics
- Model graph
- Training speed

---

## 🧪 Evaluate Your Model

After training completes:

```bash
python evaluate.py --model_path models/best_model.h5 --split test
```

**Output:**
```
Exact Match Accuracy: 0.4823
Character Accuracy: 0.7654
```

---

## 🔮 Make Predictions

### On InkML Files

```bash
python predict.py \
  --input path/to/equation.inkml \
  --model_path models/best_model.h5 \
  --visualize
```

### On Images

```bash
python predict.py \
  --input path/to/equation.png \
  --model_path models/best_model.h5 \
  --visualize
```

---

## 🎨 Visualize Dataset

Before training, explore your data:

```bash
python visualize_data.py --split train --num_samples 10
```

This shows:
- Raw strokes
- Normalized strokes
- Rendered images
- LaTeX labels

---

## 🐛 Common Issues and Solutions

### Issue 1: "Module not found" Error

**Problem**: Dependencies not installed

**Solution**:
```bash
pip install -r requirements.txt
```

### Issue 2: "Kaggle API Error"

**Problem**: Kaggle credentials not configured

**Solution**:
```bash
# Download kaggle.json from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Issue 3: "Out of Memory" Error

**Problem**: Batch size too large for your GPU

**Solution**:
```bash
python train.py --batch_size 8  # Reduce batch size
```

Or reduce model size:
```bash
python train.py --d_model 256 --num_layers 2
```

### Issue 4: Training is Very Slow

**Problem 1**: Using CPU instead of GPU

**Check GPU**:
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If no GPU found:
- Make sure you have CUDA-capable GPU
- Install CUDA toolkit
- Install cuDNN
- Reinstall TensorFlow with GPU support

**Problem 2**: Large model or batch size

**Solution**: Use smaller configuration:
```bash
python train.py --batch_size 16 --d_model 256
```

### Issue 5: "Permission Denied" on quick_start.sh

**Solution**:
```bash
chmod +x quick_start.sh
```

### Issue 6: Dataset Download Fails

**Solutions**:
1. Check internet connection
2. Verify Kaggle credentials
3. Try manual download from Kaggle website
4. Check disk space (need ~5GB)

---

## 📚 Learning Path

### Beginner Path

1. ✅ Run `test_setup.py` - Verify setup
2. ✅ Run `example_usage.py` - See it work
3. ✅ Run `visualize_data.py` - Understand data
4. ✅ Read `README.md` - Get overview
5. ✅ Train small model (10 epochs)
6. ✅ Make predictions on test files

### Intermediate Path

1. ✅ Read `ARCHITECTURE.md` - Understand model
2. ✅ Train full model (100 epochs)
3. ✅ Monitor with TensorBoard
4. ✅ Evaluate on test set
5. ✅ Experiment with hyperparameters
6. ✅ Try different model sizes

### Advanced Path

1. ✅ Read `TUTORIAL.md` - Deep dive
2. ✅ Modify model architecture
3. ✅ Implement custom loss functions
4. ✅ Add new data augmentation
5. ✅ Create ensemble of models
6. ✅ Deploy as web service

---

## 🎓 Understanding the Output

### Training Output

```
Epoch 10/100
312/312 [==============================] - 45s 144ms/step
loss: 1.2345 - masked_accuracy: 0.7234
val_loss: 1.4567 - val_masked_accuracy: 0.6543
```

**What it means:**
- `loss`: Training error (lower is better)
- `masked_accuracy`: Token-level accuracy on training set
- `val_loss`: Validation error (what you care about)
- `val_masked_accuracy`: Accuracy on validation set

**What to look for:**
- ✅ Loss should decrease over time
- ✅ Validation accuracy should increase
- ⚠️ If val_loss increases → overfitting
- ⚠️ If loss stays high → model too small or learning rate too low

### Evaluation Output

```
Exact Match Accuracy: 0.4523
Character Accuracy: 0.7834
```

**What it means:**
- `Exact Match`: % of equations predicted perfectly
- `Character Accuracy`: % of characters correct

**Typical Results:**
- After 10 epochs: 20-30% exact match
- After 50 epochs: 35-45% exact match
- After 100 epochs: 45-55% exact match

---

## 🚀 Next Steps

Once you have a working model:

### 1. Improve Accuracy
- Train longer (200 epochs)
- Use data augmentation (`--augment`)
- Increase model size (`--d_model 768`)
- Try beam search in prediction

### 2. Deploy Your Model
- Create web interface (Gradio/Streamlit)
- Build API (FastAPI/Flask)
- Mobile app (TensorFlow Lite)
- Edge device (Raspberry Pi)

### 3. Extend Functionality
- Add equation solving (SymPy)
- Support more formats
- Multi-line equations
- Handwriting style transfer

### 4. Research
- Compare with other architectures
- Try different encoders (Vision Transformer)
- Experiment with attention mechanisms
- Add pre-training

---

## 📞 Getting Help

If you're stuck:

1. **Run test script**: `python test_setup.py`
2. **Check logs**: Look in `logs/` directory
3. **Read docs**: `TUTORIAL.md` and `ARCHITECTURE.md`
4. **Common issues**: See troubleshooting section above

---

## ✅ Checklist

Before starting training, make sure:

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip list`)
- [ ] Kaggle API configured (`~/.kaggle/kaggle.json`)
- [ ] Dataset downloaded (check `data/` directory)
- [ ] Test script passes (`python test_setup.py`)
- [ ] GPU detected (optional but recommended)
- [ ] Enough disk space (~10GB for models and logs)

---

## 🎉 Success!

If you've made it this far and everything works, congratulations! 🎊

You now have a working deep learning system for handwritten equation recognition.

**Start training and watch the magic happen!** ✨

```bash
python train.py --epochs 100 --batch_size 32 --augment
```

Good luck! 🚀

