# 🔧 Troubleshooting Guide

## ⚠️ Common Issues and Solutions

---

## Issue 1: Training Errors

### **Error: `LearningRateSchedule` not settable**
```
TypeError: This optimizer was created with a `LearningRateSchedule` object...
```

**Cause**: `ReduceLROnPlateau` callback conflicts with custom learning rate schedule.

**Solution**: ✅ **FIXED** - Removed `ReduceLROnPlateau` callback from train.py

---

### **Error: `singleton tf.Variables`**
```
ValueError: tf.function only supports singleton tf.Variables created on the first call
```

**Cause**: Creating layers inside `call()` method.

**Solution**: ✅ **FIXED** - Moved Dropout layer creation to `__init__()`

---

### **Error: `get_config()` NotImplementedError**
```
NotImplementedError: Learning rate schedule 'CustomSchedule' must override `get_config()`
```

**Cause**: Custom learning rate schedule missing serialization method.

**Solution**: ✅ **FIXED** - Added `get_config()` method to CustomSchedule

---

## Issue 2: Overfitting (High Training Accuracy, Low Validation)

### **Symptoms:**
- Training accuracy: 99%+
- Validation accuracy: Much lower
- Training loss very low (< 0.05)
- Validation loss not improving

### **Example:**
```
Epoch 11: loss: 0.0119 - masked_accuracy: 0.9967
Epoch 11: val_loss: 0.0177 (not improving)
```

### **Solutions:**

#### **1. Use Data Augmentation** ⭐
```bash
python train.py --epochs 100 --batch_size 32 --augment
```

#### **2. Increase Dropout**
```bash
python train.py --dropout 0.2  # or 0.3
```

#### **3. Use Full Dataset (Not Subset)**
Make sure you're training on all 8,900 samples, not a subset.

#### **4. Early Stopping Will Help**
Already configured - will stop when validation stops improving.

#### **5. Reduce Model Complexity**
```bash
python train.py --d_model 384 --num_layers 3
```

---

## Issue 3: Training Too Slow

### **Symptoms:**
- Each epoch takes > 10 minutes
- Training will take days

### **Solutions:**

#### **1. Use Quick Training Script** ⭐
```bash
python train_quick.py  # 30-60 minutes
```

#### **2. Check GPU Usage**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If no GPU:
- Training on CPU is 10-20x slower
- Consider using Google Colab (free GPU)
- Or reduce model size

#### **3. Reduce Batch Size** (if out of memory)
```bash
python train.py --batch_size 8
```

#### **4. Use Smaller Model**
```bash
python train.py --d_model 256 --num_layers 2
```

---

## Issue 4: Out of Memory

### **Error:**
```
ResourceExhaustedError: OOM when allocating tensor
```

### **Solutions:**

#### **1. Reduce Batch Size**
```bash
python train.py --batch_size 8  # or even 4
```

#### **2. Reduce Model Size**
```bash
python train.py --d_model 256 --num_layers 2 --dff 1024
```

#### **3. Close Other Applications**
Free up RAM/GPU memory

#### **4. Use Gradient Accumulation** (advanced)
Modify train.py to accumulate gradients over multiple batches

---

## Issue 5: Model Not Improving

### **Symptoms:**
- Loss stays high (> 3.0)
- Accuracy stays low (< 10%)
- No improvement after many epochs

### **Solutions:**

#### **1. Check Learning Rate**
```bash
# Try fixed learning rate
python train.py --learning_rate 0.001
```

#### **2. Train Longer**
```bash
python train.py --epochs 200
```

#### **3. Check Data**
```bash
python visualize_data.py --num_samples 10
```
Make sure data looks correct

#### **4. Increase Model Capacity**
```bash
python train.py --d_model 768 --num_layers 6
```

---

## Issue 6: Validation Loss Increasing

### **Symptoms:**
- Training loss decreasing
- Validation loss increasing
- **This is overfitting!**

### **Solutions:**

#### **1. Use Data Augmentation** ⭐
```bash
python train.py --augment
```

#### **2. Increase Dropout**
```bash
python train.py --dropout 0.3
```

#### **3. Stop Earlier**
```bash
# Early stopping already configured with patience=15
# It will automatically stop when val_loss stops improving
```

#### **4. Use Regularization**
Add L2 regularization to model (requires code modification)

---

## Issue 7: Can't Push to GitHub

### **Error:**
```
fatal: could not read Username for 'https://github.com'
```

### **Solutions:**

#### **Option 1: GitHub CLI** ⭐
```bash
brew install gh
gh auth login
git push -u origin main
```

#### **Option 2: Personal Access Token**
```bash
# Get token from: https://github.com/settings/tokens
git remote set-url origin https://YOUR_TOKEN@github.com/USER/REPO.git
git push -u origin main
```

#### **Option 3: SSH**
```bash
# Set up SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub  # Add to GitHub

# Change remote
git remote set-url origin git@github.com:USER/REPO.git
git push -u origin main
```

---

## Issue 8: Python Version Incompatibility

### **Error:**
```
ERROR: Could not find a version that satisfies the requirement tensorflow
```

### **Solution:**
✅ **See PYTHON_COMPATIBILITY.md and QUICK_FIX.md**

You need Python 3.9-3.12 (not 3.14+)

---

## Issue 9: Kaggle API Error

### **Error:**
```
Could not find kaggle.json
```

### **Solution:**
```bash
# Download from: https://www.kaggle.com/settings
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Issue 10: Dataset Not Loading

### **Symptoms:**
- Train=0, Val=0, Test=0
- No InkML files found

### **Solutions:**

#### **1. Force Re-download**
```bash
rm -rf data/parsed_data.pkl
python src/data_loader.py
```

#### **2. Check Kaggle API**
```bash
kaggle datasets list
```

#### **3. Manual Download**
1. Go to: https://www.kaggle.com/datasets/ntcuong2103/crohme2019
2. Download manually
3. Extract to `data/` folder

---

## 🆘 Quick Diagnostic

Run this to check your setup:

```bash
python test_setup.py
```

Should show all ✓ checks passing.

---

## 📊 Expected Training Behavior

### **Normal Training:**
```
Epoch 1:  loss: 5.0 → 4.5, val_loss: 4.8
Epoch 10: loss: 3.5 → 3.0, val_loss: 3.2
Epoch 50: loss: 2.0 → 1.8, val_loss: 1.9
Epoch 100: loss: 1.5 → 1.4, val_loss: 1.5
```

### **Overfitting:**
```
Epoch 10: loss: 0.5, val_loss: 2.0  ⚠️
Epoch 20: loss: 0.1, val_loss: 2.5  ⚠️ Getting worse!
```
**Solution**: Use `--augment` and increase dropout

### **Underfitting:**
```
Epoch 50: loss: 4.0, val_loss: 4.0  ⚠️ Not learning
```
**Solution**: Increase model size or train longer

---

## 🎯 Best Practices

1. ✅ **Always use data augmentation**: `--augment`
2. ✅ **Monitor validation loss**: Should decrease
3. ✅ **Use TensorBoard**: `tensorboard --logdir logs/`
4. ✅ **Start with quick test**: `python train_quick.py`
5. ✅ **Save checkpoints**: Automatic
6. ✅ **Use early stopping**: Already configured

---

## 💡 Pro Tips

### **If training on CPU:**
- Use `train_quick.py` (smaller model)
- Or use Google Colab (free GPU)
- Expect 10-20x longer training time

### **If training on GPU:**
- Use full model: `python train.py --epochs 100 --augment`
- Monitor GPU usage: `nvidia-smi`
- Should see 80-100% GPU utilization

### **If experimenting:**
- Use `train_quick.py` for fast iterations
- Try different hyperparameters
- Use TensorBoard to compare runs

---

## 📞 Still Stuck?

1. Check `test_setup.py` - All tests passing?
2. Read `TRAINING_GUIDE.md` - Using right command?
3. Check `README.md` - Setup correct?
4. Look at logs: `tail -f logs/training_*.csv`

---

**Most issues are fixed! The project should work smoothly now.** ✅

