# 🚀 Training Guide

## ⏱️ Training Time Expectations

### **Full Model (Production Quality)**
```bash
python train.py --epochs 100 --batch_size 32 --augment
```
- **Time**: 8-12 hours (GPU) or 50-100 hours (CPU)
- **Model size**: ~25M parameters
- **Expected accuracy**: 45-55% exact match
- **Best for**: Final production model

---

### **Quick Test Model (Recommended for Testing)** ⭐
```bash
python train_quick.py
```
- **Time**: 30-60 minutes
- **Model size**: ~1.5M parameters  
- **Configuration**: 10 epochs, smaller model (256 dim, 2 layers)
- **Expected accuracy**: 20-30% exact match
- **Best for**: Quick testing, debugging, experimentation

---

### **Mini Test (Fastest)**
```bash
python train.py --epochs 5 --batch_size 8 --d_model 128 --num_layers 1
```
- **Time**: 10-15 minutes
- **Model size**: ~500K parameters
- **Expected accuracy**: 10-15% exact match
- **Best for**: Verifying setup works

---

## 🎯 Which Should You Use?

### **If you want to:**

**✅ Test that everything works**
→ Run `python train_quick.py` (30-60 min)

**✅ Get production model**  
→ Run full training overnight: `python train.py --epochs 100 --batch_size 32 --augment`

**✅ Quick experiment with hyperparameters**
→ Use mini test with custom params

---

## 🔧 Common Training Commands

### **Resume from checkpoint:**
```bash
python train.py --resume models/model_*.h5 --epochs 100
```

### **Custom configuration:**
```bash
python train.py \
  --epochs 50 \
  --batch_size 16 \
  --d_model 384 \
  --num_layers 3 \
  --learning_rate 0.0001
```

### **With data augmentation:**
```bash
python train.py --epochs 100 --augment
```

---

## 📊 Monitoring Training

### **Option 1: Watch console output**
Training will show progress bars and metrics:
```
Epoch 1/100
279/279 ━━━━━━━━━━━━━━━━━━━━ 45s - loss: 3.61 - val_loss: 1.79
```

### **Option 2: TensorBoard (Recommended)**
Open a new terminal:
```bash
cd /path/to/project
source .venv/bin/activate
tensorboard --logdir logs/
```
Then open: http://localhost:6006

### **Option 3: Check logs**
```bash
tail -f logs/training_run_*.csv
```

---

## ⚡ Speed Tips

### **Make it faster:**
1. **Use GPU** - 10-20x faster than CPU
2. **Reduce batch size** - If running out of memory
3. **Use smaller model** - Reduce d_model, num_layers
4. **Train overnight** - Set it and forget it

### **Check if using GPU:**
```bash
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

---

## 🐛 Troubleshooting

### **Issue: Out of Memory**
**Solution**: Reduce batch size or model size
```bash
python train.py --batch_size 8 --d_model 256
```

### **Issue: Training too slow**
**Solutions**:
1. Use `train_quick.py` instead
2. Check if GPU is being used
3. Reduce number of samples (for testing)

### **Issue: Model not improving**
**Solutions**:
1. Train longer (50-100 epochs minimum)
2. Use data augmentation: `--augment`
3. Try different learning rate: `--learning_rate 0.001`

### **Issue: Model saving error**
**Fixed!** The `get_config()` method has been added to CustomSchedule.

---

## 📈 Expected Progress

### **Epoch 1-10:**
- Loss: 5.0 → 3.5
- Accuracy: 0% → 15%
- Learning basic patterns

### **Epoch 10-50:**
- Loss: 3.5 → 2.0
- Accuracy: 15% → 35%
- Learning structure

### **Epoch 50-100:**
- Loss: 2.0 → 1.5
- Accuracy: 35% → 50%
- Fine-tuning

---

## 💡 Pro Tips

1. **Start with quick test** - Use `train_quick.py` first
2. **Monitor validation loss** - Should decrease over time
3. **Use early stopping** - Already configured (patience=15)
4. **Save checkpoints** - Automatically saved to `models/`
5. **Train overnight** - Use `nohup` or `screen`:
   ```bash
   nohup python train.py --epochs 100 --batch_size 32 > training.log 2>&1 &
   ```

---

## 🎓 Understanding the Output

```
Epoch 50/100
279/279 ━━━━━━━━━━━━━━━━━━━━ 45s 161ms/step
loss: 2.1234 - masked_accuracy: 0.3456
val_loss: 1.9876 - val_masked_accuracy: 0.4123
```

**What it means:**
- `279/279` - Number of batches processed
- `loss` - Training error (lower is better)
- `masked_accuracy` - Training accuracy (higher is better)
- `val_loss` - Validation error (what matters most)
- `val_masked_accuracy` - Validation accuracy

**Good signs:**
- ✅ val_loss decreasing
- ✅ val_accuracy increasing
- ✅ No huge gap between train and val

**Bad signs:**
- ⚠️ val_loss increasing (overfitting)
- ⚠️ Loss staying flat (learning rate too low)
- ⚠️ Huge train/val gap (overfitting)

---

## 🚀 Quick Start Recommendation

**For first time:**
```bash
# 1. Quick test (30-60 min)
python train_quick.py

# 2. If it works, start full training
nohup python train.py --epochs 100 --batch_size 32 --augment > training.log 2>&1 &

# 3. Monitor in another terminal
tail -f training.log
# Or use TensorBoard
```

---

## ⏰ Time Budget

| Task | Time | Command |
|------|------|---------|
| Quick test | 30-60 min | `python train_quick.py` |
| Medium training | 2-4 hours | `python train.py --epochs 30` |
| Full training (CPU) | 2-3 days | `python train.py --epochs 100` |
| Full training (GPU) | 8-12 hours | `python train.py --epochs 100` |

---

**Bottom line**: Use `python train_quick.py` for testing, then run full training overnight if you want production quality! 🎯

