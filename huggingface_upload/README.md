# Handwritten Equation Solver - CNN-Transformer Model

## Model Description

This model solves handwritten mathematical equations by converting stroke data into LaTeX format. It uses a hybrid **CNN-Transformer architecture** trained on the CROHME2019 dataset.

### Architecture
- **Encoder**: CNN-based feature extractor
  - Convolutional blocks for spatial feature extraction
  - Processes 128x128 grayscale images
- **Decoder**: Transformer-based sequence generator
  - Multi-head self-attention mechanism
  - Generates LaTeX token sequences
- **Model Size**: ~1.5M parameters (Quick model)
- **Input**: 128x128 grayscale images of handwritten equations
- **Output**: LaTeX string representation

## Training Details

### Dataset
- **Source**: CROHME2019 (Competition on Recognition of Online Handwritten Mathematical Expressions)
- **Training Samples**: 8,900
- **Validation Samples**: 986
- **Test Samples**: 1,199

### Training Configuration
- **Epochs**: 10 (Quick training)
- **Batch Size**: 16
- **Model Dimension**: 256
- **Transformer Layers**: 2
- **Attention Heads**: 4
- **Feed-forward Dimension**: 1024
- **Dropout**: 0.1
- **Learning Rate**: Custom warmup schedule (d_model-based)
- **Optimizer**: Adam
- **Data Augmentation**: 
  - Random rotation (±15°)
  - Random scaling (0.9-1.1)
  - Random translation (±10%)

### Performance
- **Training Accuracy**: 99.86%
- **Validation Accuracy**: 99.77%
- **Best Epoch**: 5 (val_loss: 0.0186)

## Usage

### Installation

```bash
pip install tensorflow>=2.15.0 keras>=2.15.0 numpy opencv-python
```

**Note**: This model requires Python 3.9-3.12. TensorFlow does not support Python 3.13+ yet.

### Loading the Model

```python
import tensorflow as tf
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="YOUR_USERNAME/handwritten-equation-solver",
    filename="best_model.keras"
)

# Load model
model = tf.keras.models.load_model(model_path, compile=False)

# Download vocabulary
vocab_path = hf_hub_download(
    repo_id="YOUR_USERNAME/handwritten-equation-solver",
    filename="vocabulary.pkl"
)
```

### Making Predictions

```python
import numpy as np
import pickle

# Load vocabulary
with open(vocab_path, 'rb') as f:
    vocabulary = pickle.load(f)

# Prepare your image (128x128 grayscale, normalized to [0,1])
# Example: Load and preprocess your handwritten equation image
from PIL import Image
import cv2

# Load image
img = Image.open("your_equation.png").convert('L')
img = img.resize((128, 128))
image = np.array(img, dtype=np.float32) / 255.0
image = np.expand_dims(image, axis=-1)  # Add channel dimension
image = np.expand_dims(image, axis=0)   # Add batch dimension

# Create decoder input (start with <SOS> token)
decoder_input = np.array([[vocabulary.char2idx['<SOS>']]])

# Generate prediction (autoregressive)
max_length = 100
for _ in range(max_length):
    predictions = model.predict([image, decoder_input], verbose=0)
    predicted_id = np.argmax(predictions[0, -1, :])
    
    if predicted_id == vocabulary.char2idx['<EOS>']:
        break
    
    decoder_input = np.concatenate([
        decoder_input, 
        [[predicted_id]]
    ], axis=1)

# Decode to LaTeX
latex_output = vocabulary.decode(decoder_input[0])
print(f"Predicted LaTeX: {latex_output}")
```

## Model Files

- `best_model.keras`: Trained model weights (best validation loss)
- `vocabulary.pkl`: Token vocabulary for encoding/decoding
- `training_history.csv`: Training metrics log
- `config.json`: Model configuration

## Limitations

- Trained specifically on CROHME2019 equation style
- Limited to symbols in the training vocabulary
- Best performance on equations similar to training data
- Requires properly normalized input images (128x128)
- High accuracy might indicate overfitting on this specific dataset

## Training Code

The complete training code and dataset preparation scripts are available at:
[GitHub Repository](https://github.com/YOUR_USERNAME/DeepLearning-Project-Handwritten-Equation-Solver)

## Citation

If you use this model, please cite the CROHME2019 dataset:

```bibtex
@inproceedings{crohme2019,
  title={CROHME 2019: Competition on Recognition of Online Handwritten Mathematical Expressions},
  author={Mahdavi, Mahshad and Condon, Richard and Davila, Kenny and Zanibbi, Richard},
  booktitle={International Conference on Document Analysis and Recognition},
  year={2019}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on the GitHub repository.

---

**Model Type**: Sequence-to-Sequence (Image-to-Text)  
**Language**: LaTeX (Mathematical Notation)  
**Framework**: TensorFlow/Keras  
**Dataset**: CROHME2019  
**Task**: Handwritten Equation Recognition

