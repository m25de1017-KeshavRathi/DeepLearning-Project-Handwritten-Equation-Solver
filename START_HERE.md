# ⚠️ START HERE - Important Setup Information

## 🛑 READ THIS FIRST

**You have Python 3.14 installed.**  
**This project requires Python 3.9-3.12 because TensorFlow doesn't support Python 3.14 yet.**

---

## ✅ What You Need To Do

### Step 1: Install Python 3.12

Choose your operating system:

#### **macOS** 🍎
```bash
brew install python@3.12
```

#### **Ubuntu/Debian** 🐧
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv
```

#### **Windows** 🪟
Download from: https://www.python.org/downloads/release/python-3120/

---

### Step 2: Create Virtual Environment with Python 3.12

```bash
# Navigate to project directory
cd /Users/chiragphor/Development/projects/DeepLearning-Project-Handwritten-Equation-Solver

# Remove old virtual environment (if it exists)
rm -rf venv

# Create new virtual environment with Python 3.12
python3.12 -m venv venv

# Activate it
source venv/bin/activate

# Verify Python version (should show 3.12.x)
python --version
```

---

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**This should work now!** ✅

---

### Step 4: Verify Installation

```bash
# Test TensorFlow
python -c "import tensorflow as tf; print(f'✅ TensorFlow {tf.__version__}')"

# Run full test
python test_setup.py
```

**Expected output:**
```
✅ TensorFlow 2.15.0 (or later)
✅ All tests passed!
```

---

### Step 5: Try the Demo

```bash
python example_usage.py
```

---

## 📊 Python Version Compatibility

| Python Version | Status | Can Use? |
|---------------|--------|----------|
| 3.8 | ⚠️ Old | Maybe |
| 3.9 | ✅ Supported | Yes ⭐ |
| 3.10 | ✅ Supported | Yes ⭐ |
| 3.11 | ✅ Supported | Yes ⭐ |
| 3.12 | ✅ Supported | Yes ⭐⭐⭐ |
| 3.13 | ❌ Too new | No |
| 3.14 | ❌ Too new | No |

**Recommendation: Use Python 3.12** 🎯

---

## 🔍 Why This Happens

TensorFlow (the deep learning library) is one of the most complex Python packages. It:
- Has compiled C++ code
- Interfaces with CUDA (GPU acceleration)
- Requires extensive testing for each Python version

**It takes time for TensorFlow to support new Python versions.**

Python 3.14 is very new (released Oct 2024), and TensorFlow hasn't caught up yet.

---

## 💡 Quick Commands

**All-in-one setup (after installing Python 3.12):**

```bash
cd /Users/chiragphor/Development/projects/DeepLearning-Project-Handwritten-Equation-Solver && \
rm -rf venv && \
python3.12 -m venv venv && \
source venv/bin/activate && \
pip install --upgrade pip && \
pip install -r requirements.txt && \
python test_setup.py && \
echo "✅ Setup complete! Run: python example_usage.py"
```

---

## 📚 More Information

- **Quick fix guide**: [QUICK_FIX.md](QUICK_FIX.md)
- **Detailed compatibility info**: [PYTHON_COMPATIBILITY.md](PYTHON_COMPATIBILITY.md)
- **Getting started guide**: [GETTING_STARTED.md](GETTING_STARTED.md)
- **Full tutorial**: [TUTORIAL.md](TUTORIAL.md)

---

## ❓ Still Having Issues?

### Check Python Version
```bash
python --version
```
Should show: `Python 3.12.x`

### Check Virtual Environment is Activated
Your terminal should show: `(venv)` at the beginning of the line

### Still Stuck?
1. Delete `venv` folder: `rm -rf venv`
2. Start from Step 2 above
3. Make sure you're using `python3.12` (not just `python` or `python3`)

---

## 🎯 Bottom Line

1. **Install Python 3.12** (5 minutes)
2. **Create fresh virtual environment** with Python 3.12 (1 minute)
3. **Install dependencies** (5-10 minutes)
4. **Start building!** 🚀

---

**Once setup is complete, you'll have a powerful deep learning system for recognizing handwritten equations!** 🎉

Continue to [GETTING_STARTED.md](GETTING_STARTED.md) after successful setup.

