# Python Version Compatibility Guide

## ⚠️ Important: Python Version Requirements

**TensorFlow has specific Python version requirements!**

---

## ✅ Recommended Python Versions

| Python Version | TensorFlow Support | Status | Recommendation |
|---------------|-------------------|---------|----------------|
| **3.9** | ✅ Full support | Stable | ⭐ Recommended |
| **3.10** | ✅ Full support | Stable | ⭐ Recommended |
| **3.11** | ✅ Full support | Stable | ⭐ Recommended |
| **3.12** | ✅ TF 2.16+ | Stable | ⭐ Recommended |
| **3.13+** | ⚠️ Limited | Beta | Use nightly builds |
| **3.14+** | ❌ Not supported | Unstable | Not recommended |

---

## 🔧 Solutions for Python 3.14+

You have **Python 3.14**, which is too new for stable TensorFlow. Here are your options:

### Option 1: Install Python 3.11 or 3.12 (Recommended)

**Why**: Most stable and well-tested option.

#### On macOS (using Homebrew):
```bash
# Install Python 3.12
brew install python@3.12

# Create virtual environment with Python 3.12
python3.12 -m venv venv

# Activate it
source venv/bin/activate

# Verify version
python --version  # Should show Python 3.12.x

# Install dependencies
pip install -r requirements.txt
```

#### On Ubuntu/Debian:
```bash
# Add deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.12
sudo apt install python3.12 python3.12-venv

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### On Windows:
1. Download Python 3.12 from [python.org](https://www.python.org/downloads/)
2. Install it
3. Create virtual environment:
```bash
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

### Option 2: Use TensorFlow Nightly Builds

**Why**: Bleeding edge, may have Python 3.14 support.

**Warning**: Nightly builds are unstable and may have bugs!

```bash
# Create/activate virtual environment
python3.14 -m venv venv
source venv/bin/activate

# Install TensorFlow nightly
pip install tf-nightly

# Install other dependencies
pip install numpy opencv-python matplotlib scikit-learn pandas pillow kagglehub lxml tqdm keras
```

---

### Option 3: Use pyenv (Advanced)

**Why**: Manage multiple Python versions easily.

```bash
# Install pyenv (macOS)
brew install pyenv

# Or (Linux)
curl https://pyenv.run | bash

# Install Python 3.12
pyenv install 3.12.0

# Set as local version for this project
cd /path/to/project
pyenv local 3.12.0

# Create virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🧪 Testing Your Setup

After setting up the correct Python version:

```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Test imports
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"

# Run test script
python test_setup.py
```

**Expected output:**
```
TensorFlow 2.15.0 (or later)
✓ All packages imported successfully!
```

---

## 📋 Quick Fix Checklist

If you're getting TensorFlow installation errors:

1. **Check Python version**:
   ```bash
   python --version
   ```

2. **If 3.14+, choose one of the options above**

3. **Create fresh virtual environment** with correct Python:
   ```bash
   # Remove old venv if it exists
   rm -rf venv
   
   # Create new one with Python 3.12
   python3.12 -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Verify installation**:
   ```bash
   python test_setup.py
   ```

---

## 🔍 Troubleshooting

### Error: "Could not find a version that satisfies the requirement tensorflow"

**Cause**: Your Python version is not supported by TensorFlow.

**Solution**: Use Python 3.9-3.12 (see Option 1 above).

---

### Error: "No matching distribution found for tensorflow"

**Cause**: Either Python version issue or network problem.

**Solutions**:
1. Check Python version: `python --version`
2. Try upgrading pip: `pip install --upgrade pip`
3. Try specific version: `pip install tensorflow==2.15.0`
4. Check internet connection

---

### Error: Module 'tensorflow' has no attribute 'X'

**Cause**: Using TensorFlow nightly with breaking changes.

**Solution**: Switch to stable Python version (3.9-3.12).

---

## 🎯 Recommended Setup

**For this project, we recommend:**

✅ **Python 3.11** or **3.12**  
✅ **TensorFlow 2.15.0** or later  
✅ **Virtual environment**  
✅ **Fresh installation**  

**Setup commands:**
```bash
# 1. Install Python 3.12 (if needed)
brew install python@3.12  # macOS

# 2. Create project virtual environment
cd /Users/chiragphor/Development/projects/DeepLearning-Project-Handwritten-Equation-Solver
python3.12 -m venv venv

# 3. Activate
source venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Test
python test_setup.py

# 6. Run demo
python example_usage.py
```

---

## 📊 Python Version Feature Support

| Feature | Python 3.9 | Python 3.10 | Python 3.11 | Python 3.12 | Python 3.14 |
|---------|-----------|------------|------------|------------|------------|
| TensorFlow Stable | ✅ | ✅ | ✅ | ✅ | ❌ |
| All packages work | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| Performance | Good | Good | Better | Best | Best |
| Stability | High | High | High | High | Low |
| **Recommendation** | ⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ❌ |

---

## 🚀 Next Steps

Once you have the correct Python version:

1. ✅ Run `python test_setup.py`
2. ✅ Run `python example_usage.py`
3. ✅ Start training: `python train.py --epochs 100 --batch_size 32`

---

## 💡 Pro Tips

1. **Always use virtual environments** - Keeps dependencies isolated
2. **Stick to Python 3.11 or 3.12** - Best compatibility
3. **Don't use bleeding-edge Python** - Wait for library support
4. **Keep pip updated** - `pip install --upgrade pip`
5. **Check compatibility first** - Before upgrading Python

---

## 📞 Still Having Issues?

If you're still stuck:

1. **Delete old virtual environment**: `rm -rf venv`
2. **Install Python 3.12**: Follow Option 1 above
3. **Create fresh venv**: `python3.12 -m venv venv`
4. **Try again**: `source venv/bin/activate && pip install -r requirements.txt`
5. **Run test**: `python test_setup.py`

---

**Bottom Line**: For the best experience with this project, use Python 3.11 or 3.12! 🎯

