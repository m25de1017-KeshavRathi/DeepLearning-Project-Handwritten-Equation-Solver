# 🤗 Upload Model to Hugging Face

## Quick Start

### Step 1: Get Your Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "model-upload")
4. Select "Write" permissions
5. Copy the token

### Step 2: Upload Model

```bash
# Activate virtual environment
source .venv/bin/activate

# Run upload script
python upload_to_huggingface.py
```

The script will:
1. Ask for your Hugging Face token
2. Ask for your username
3. Create a repository (default: `your-username/handwritten-equation-solver`)
4. Upload all files from `huggingface_upload/` folder

### Step 3: Verify Upload

After upload completes, visit:
```
https://huggingface.co/YOUR_USERNAME/handwritten-equation-solver
```

---

## Custom Repository Name

```bash
python upload_to_huggingface.py --repo_name my-custom-name
```

---

## Files Being Uploaded

The following files will be uploaded to Hugging Face:

| File | Size | Description |
|------|------|-------------|
| `best_model.keras` | ~56 MB | Trained model weights (best validation loss) |
| `vocabulary.pkl` | 1.3 KB | Token vocabulary for encoding/decoding |
| `README.md` | 4.4 KB | Model card with usage instructions |
| `config.json` | 845 B | Model configuration and metadata |

**Total size: ~56 MB**

---

## Using Your Model from Hugging Face

After upload, anyone can use your model:

```python
import tensorflow as tf
from huggingface_hub import hf_hub_download
import pickle
import numpy as np

# Download model
model_path = hf_hub_download(
    repo_id="YOUR_USERNAME/handwritten-equation-solver",
    filename="best_model.keras"
)

# Download vocabulary
vocab_path = hf_hub_download(
    repo_id="YOUR_USERNAME/handwritten-equation-solver",
    filename="vocabulary.pkl"
)

# Load model
model = tf.keras.models.load_model(model_path, compile=False)

# Load vocabulary
with open(vocab_path, 'rb') as f:
    vocabulary = pickle.load(f)

# Use for predictions
# (See README.md in the uploaded model for full usage)
```

---

## Troubleshooting

### Token Invalid
- Make sure you selected "Write" permissions when creating the token
- Copy the entire token without extra spaces

### Network Issues
- Check your internet connection
- Try again with `--trusted-host pypi.org` if SSL issues occur

### Upload Fails
- Verify files exist in `huggingface_upload/` folder
- Check file sizes aren't corrupted
- Ensure you're logged in to Hugging Face

---

## Alternative: Manual Upload

If the script doesn't work, you can upload manually:

1. Go to https://huggingface.co/new
2. Create a new model repository
3. Click "Files and versions" → "Add file" → "Upload files"
4. Upload all files from `huggingface_upload/` folder
5. Done!

---

## Making Repository Private

To make your model private:

```python
from huggingface_hub import update_repo_visibility

update_repo_visibility(
    repo_id="YOUR_USERNAME/handwritten-equation-solver",
    private=True
)
```

Or change it on the website:
- Go to your model page
- Click "Settings"
- Toggle "Private"

---

**Ready to upload? Run `python upload_to_huggingface.py`!** 🚀

