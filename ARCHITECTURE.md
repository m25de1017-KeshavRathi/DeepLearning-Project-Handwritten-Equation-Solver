# Model Architecture Documentation

## Overview

The Handwritten Equation Solver uses a **CNN-Transformer** hybrid architecture that combines the strengths of convolutional neural networks for visual feature extraction with transformer networks for sequence generation.

---

## Architecture Components

### 1. CNN Encoder

**Purpose**: Extract visual features from handwritten mathematical expressions

**Architecture**:
```
Input Image (128×128×1)
    ↓
Conv2D(64, 3×3) + BatchNorm + ReLU + MaxPool(2×2) → 64×64×64
    ↓
Conv2D(128, 3×3) + BatchNorm + ReLU + MaxPool(2×2) → 32×32×128
    ↓
Conv2D(256, 3×3) + BatchNorm + ReLU + MaxPool(2×2) → 16×16×256
    ↓
Conv2D(512, 3×3) + BatchNorm + ReLU + MaxPool(2×2) → 8×8×512
    ↓
Conv2D(512, 3×3) + BatchNorm + ReLU → 8×8×512
    ↓
Reshape → (64, 512)  # Sequence of feature vectors
```

**Key Features**:
- Progressive feature extraction through 5 convolutional layers
- Batch normalization for training stability
- Dropout (0.2) for regularization
- Output is a sequence of 512-dimensional feature vectors

**Design Rationale**:
- CNNs excel at capturing spatial hierarchies in images
- Multiple scales capture both fine strokes and overall structure
- Sequence output enables attention mechanism

---

### 2. Transformer Decoder

**Purpose**: Generate LaTeX sequence from visual features

**Architecture**:

```
Encoder Features (64×512)     Target Sequence (padded to max_len)
         ↓                              ↓
         |                    Embedding + Positional Encoding
         |                              ↓
         |                    ┌──────────────────┐
         |                    │  Decoder Layer 1  │
         |                    │  ├─ Self-Attention │
         └────────────────────┤  ├─ Cross-Attention│
                              │  └─ Feed-Forward   │
                              └──────────────────┘
                                       ↓
                              ┌──────────────────┐
                              │  Decoder Layer 2  │
                              │  ├─ Self-Attention │
                              ├─ Cross-Attention  │
                              │  └─ Feed-Forward   │
                              └──────────────────┘
                                       ↓
                              ... (repeat for N layers)
                                       ↓
                              Dense(vocab_size)
                                       ↓
                              Softmax → Predictions
```

**Decoder Layer Details**:

Each decoder layer consists of:

1. **Masked Self-Attention**
   - Attends to previous positions in target sequence
   - Prevents looking ahead during training
   - Multi-head attention (8 heads)
   - Dimension: 512, Key/Query/Value: 64 per head

2. **Cross-Attention**
   - Attends to encoder output (visual features)
   - Learns alignment between strokes and symbols
   - Multi-head attention (8 heads)

3. **Feed-Forward Network**
   - Two-layer MLP: 512 → 2048 → 512
   - ReLU activation
   - Dropout (0.1)

4. **Layer Normalization**
   - Applied after each sub-layer
   - Stabilizes training

5. **Residual Connections**
   - Skip connections around each sub-layer
   - Enables deep network training

---

## Mathematical Formulation

### Attention Mechanism

**Scaled Dot-Product Attention**:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- Q (Query): What we're looking for
- K (Key): What we're comparing against
- V (Value): What we return
- d_k: Dimension of keys (64)

**Multi-Head Attention**:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

Benefits:
- Captures different types of relationships
- Increases model capacity
- Enables parallel computation

### Positional Encoding

Since transformers have no inherent notion of sequence order:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- pos: Position in sequence
- i: Dimension index

This encoding:
- Is deterministic (no learned parameters)
- Allows model to learn relative positions
- Works for sequences of any length

---

## Training

### Loss Function

**Masked Cross-Entropy Loss**:

```
L = -∑∑ mask(i,j) · log P(y_j^(i) | y_<j^(i), x^(i))
     i j
```

Where:
- i: Sample index
- j: Position in sequence
- mask: Ignores padding tokens
- y_<j: Previous tokens in sequence
- x: Input image

### Optimization

**Adam Optimizer** with custom learning rate schedule:

```
lr(step) = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))
```

This schedule:
- Increases learning rate during warmup (first 4000 steps)
- Decreases afterwards with inverse square root
- Stabilizes training of deep transformers

**Parameters**:
- β₁ = 0.9
- β₂ = 0.98
- ε = 10⁻⁹

---

## Inference

### Greedy Decoding

Simple, fast approach:

```python
def greedy_decode(model, image):
    encoder_output = model.encode(image)
    sequence = [START_TOKEN]
    
    while len(sequence) < MAX_LEN:
        predictions = model.decode(sequence, encoder_output)
        next_token = argmax(predictions[-1])
        sequence.append(next_token)
        
        if next_token == END_TOKEN:
            break
    
    return sequence
```

**Pros**: Fast, simple
**Cons**: May miss better sequences

### Beam Search

Better quality, slower:

```python
def beam_search(model, image, beam_width=5):
    encoder_output = model.encode(image)
    beams = [([START_TOKEN], 0.0)]  # (sequence, log_prob)
    
    for _ in range(MAX_LEN):
        candidates = []
        
        for sequence, score in beams:
            if sequence[-1] == END_TOKEN:
                candidates.append((sequence, score))
                continue
            
            predictions = model.decode(sequence, encoder_output)
            top_k = get_top_k(predictions[-1], beam_width)
            
            for token, log_prob in top_k:
                new_seq = sequence + [token]
                new_score = score + log_prob
                candidates.append((new_seq, new_score))
        
        beams = sorted(candidates, key=lambda x: x[1])[:beam_width]
    
    return beams[0][0]  # Best sequence
```

**Pros**: Better quality
**Cons**: Slower (beam_width × slower)

---

## Model Variants

### Small Model (Fast Training)
- d_model: 256
- num_heads: 4
- num_layers: 2
- dff: 1024
- Parameters: ~5M

### Default Model (Balanced)
- d_model: 512
- num_heads: 8
- num_layers: 4
- dff: 2048
- Parameters: ~25M

### Large Model (Best Accuracy)
- d_model: 768
- num_heads: 12
- num_layers: 6
- dff: 3072
- Parameters: ~80M

---

## Data Flow

### Training

```
1. Load InkML file
   ↓
2. Extract strokes (list of (x,y) coordinates)
   ↓
3. Normalize strokes (scale to [0,1])
   ↓
4. Apply augmentation (rotation, scaling, translation)
   ↓
5. Render to image (128×128)
   ↓
6. Encode label to token indices
   ↓
7. Feed to model
   ↓
8. Calculate loss (masked cross-entropy)
   ↓
9. Backpropagate and update weights
```

### Inference

```
1. Load InkML or image
   ↓
2. Normalize and render to 128×128
   ↓
3. Encode with CNN
   ↓
4. Decode with Transformer (greedy or beam search)
   ↓
5. Convert token indices to LaTeX string
   ↓
6. Return prediction
```

---

## Key Design Decisions

### Why CNN Encoder?
- **Spatial invariance**: CNNs handle stroke variations well
- **Efficiency**: Parameter sharing reduces model size
- **Hierarchy**: Multiple scales capture different features
- **Proven**: CNNs excel at image tasks

### Why Transformer Decoder?
- **Long-range dependencies**: Mathematical expressions have complex structure
- **Attention**: Explicitly models alignment between input and output
- **Parallelization**: Faster training than RNNs
- **State-of-the-art**: Best results in sequence-to-sequence tasks

### Why Not End-to-End CNN?
- CNNs struggle with sequential output
- Limited receptive field for long sequences
- No explicit alignment mechanism

### Why Not RNN Decoder?
- Transformers train faster (parallel vs. sequential)
- Better at long-range dependencies
- No vanishing gradient issues
- Higher accuracy on benchmarks

---

## Performance Characteristics

### Time Complexity

**Training**:
- Encoder: O(H·W·d_model)
- Decoder: O(L²·d_model) for self-attention
- Cross-attention: O(L·S·d_model) where S = sequence length
- Per batch: O(B·(HW + L²)·d_model)

**Inference**:
- Greedy: O(L·(HW + L)·d_model)
- Beam search: O(B·L·(HW + L)·d_model)

### Memory Complexity

**Training**:
- Model parameters: ~25M parameters × 4 bytes = 100MB
- Activations: ~500MB per batch (batch_size=32)
- Gradients: Same as parameters
- **Total**: ~1.5GB (GPU memory)

**Inference**:
- Model: 100MB
- Single sample: ~50MB
- **Total**: ~200MB

---

## Extending the Architecture

### Possible Improvements

1. **Vision Transformer Encoder**
   - Replace CNN with ViT
   - Better at global context
   - Requires more data

2. **Relative Position Encodings**
   - Better for variable-length sequences
   - Learned position bias

3. **Pre-trained Encoder**
   - Use ImageNet pre-trained CNN
   - Faster convergence
   - Better generalization

4. **Hierarchical Attention**
   - Multi-scale attention
   - Better for complex expressions

5. **Reinforcement Learning**
   - Optimize for sequence-level metrics
   - Better at avoiding errors

---

## References

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Original Transformer architecture

2. **CROHME Competition** (Mahdavi et al., 2019)
   - Dataset and benchmark

3. **Image Transformer** (Parmar et al., 2018)
   - Transformers for images

4. **CNN-RNN for Handwriting Recognition** (Zhang et al., 2017)
   - Hybrid architectures

---

This architecture strikes a balance between:
- **Accuracy**: State-of-the-art results on CROHME
- **Efficiency**: Trainable on single GPU
- **Flexibility**: Easy to adapt to related tasks
- **Interpretability**: Attention weights show alignment

