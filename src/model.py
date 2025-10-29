"""
Deep learning model architecture for handwritten equation recognition.
Uses CNN encoder + Transformer decoder architecture.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple


class PositionalEncoding(layers.Layer):
    """Positional encoding layer for transformer."""
    
    def __init__(self, max_len: int = 100, d_model: int = 512):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
    
    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]


class CNNEncoder(layers.Layer):
    """CNN encoder for extracting visual features from handwritten images."""
    
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        
        # CNN layers
        self.conv1 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(2)
        
        self.conv2 = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D(2)
        
        self.conv3 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.bn3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D(2)
        
        self.conv4 = layers.Conv2D(512, 3, padding='same', activation='relu')
        self.bn4 = layers.BatchNormalization()
        self.pool4 = layers.MaxPooling2D(2)
        
        self.conv5 = layers.Conv2D(feature_dim, 3, padding='same', activation='relu')
        self.bn5 = layers.BatchNormalization()
        
        self.dropout = layers.Dropout(0.2)
    
    def call(self, x, training=False):
        # Input shape: (batch, height, width, 1)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        x = self.dropout(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        x = self.dropout(x, training=training)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)
        x = self.dropout(x, training=training)
        
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.pool4(x)
        x = self.dropout(x, training=training)
        
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        
        # Reshape to sequence: (batch, height*width, feature_dim)
        batch_size = tf.shape(x)[0]
        h, w = tf.shape(x)[1], tf.shape(x)[2]
        x = tf.reshape(x, (batch_size, h * w, self.feature_dim))
        
        return x


class TransformerDecoder(layers.Layer):
    """Transformer decoder for generating LaTeX sequences."""
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dff: int = 2048,
                 max_len: int = 100,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_len, d_model)
        
        # Transformer decoder layers
        self.decoder_layers = [
            layers.MultiHeadAttention(num_heads, d_model // num_heads, dropout=dropout_rate)
            for _ in range(num_layers)
        ]
        
        self.cross_attention_layers = [
            layers.MultiHeadAttention(num_heads, d_model // num_heads, dropout=dropout_rate)
            for _ in range(num_layers)
        ]
        
        self.ffn_layers = []
        for _ in range(num_layers):
            ffn = keras.Sequential([
                layers.Dense(dff, activation='relu'),
                layers.Dropout(dropout_rate),
                layers.Dense(d_model)
            ])
            self.ffn_layers.append(ffn)
        
        self.layernorm1_layers = [layers.LayerNormalization() for _ in range(num_layers)]
        self.layernorm2_layers = [layers.LayerNormalization() for _ in range(num_layers)]
        self.layernorm3_layers = [layers.LayerNormalization() for _ in range(num_layers)]
        
        self.dropout_layers = [layers.Dropout(dropout_rate) for _ in range(num_layers)]
        
        # Output layer
        self.final_layer = layers.Dense(vocab_size)
    
    def create_look_ahead_mask(self, size):
        """Create look-ahead mask to prevent attending to future positions."""
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask
    
    def call(self, x, encoder_output, training=False):
        # x shape: (batch, seq_len)
        seq_len = tf.shape(x)[1]
        
        # Create look-ahead mask
        look_ahead_mask = self.create_look_ahead_mask(seq_len)
        
        # Embedding and positional encoding
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = layers.Dropout(0.1)(x, training=training)
        
        # Transformer decoder layers
        for i in range(self.num_layers):
            # Self-attention
            attn_output = self.decoder_layers[i](
                x, x, attention_mask=look_ahead_mask, training=training
            )
            attn_output = self.dropout_layers[i](attn_output, training=training)
            x = self.layernorm1_layers[i](x + attn_output)
            
            # Cross-attention
            cross_attn_output = self.cross_attention_layers[i](
                x, encoder_output, training=training
            )
            cross_attn_output = self.dropout_layers[i](cross_attn_output, training=training)
            x = self.layernorm2_layers[i](x + cross_attn_output)
            
            # Feed forward network
            ffn_output = self.ffn_layers[i](x, training=training)
            ffn_output = self.dropout_layers[i](ffn_output, training=training)
            x = self.layernorm3_layers[i](x + ffn_output)
        
        # Final output layer
        output = self.final_layer(x)  # (batch, seq_len, vocab_size)
        
        return output


class HandwrittenEquationSolver(keras.Model):
    """Complete model for handwritten equation recognition."""
    
    def __init__(self,
                 vocab_size: int,
                 img_size: Tuple[int, int] = (128, 128),
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_decoder_layers: int = 4,
                 dff: int = 2048,
                 max_len: int = 100,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Encoder
        self.encoder = CNNEncoder(d_model)
        
        # Decoder
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            dff=dff,
            max_len=max_len,
            dropout_rate=dropout_rate
        )
    
    def call(self, inputs, training=False):
        # inputs is a dictionary with 'input_image' and 'input_label'
        image = inputs['input_image']
        label = inputs['input_label']
        
        # Encode image
        encoder_output = self.encoder(image, training=training)
        
        # Decode to generate sequence
        decoder_output = self.decoder(label, encoder_output, training=training)
        
        return decoder_output
    
    def get_config(self):
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model
        }


def create_model(vocab_size: int,
                img_size: Tuple[int, int] = (128, 128),
                d_model: int = 512,
                num_heads: int = 8,
                num_decoder_layers: int = 4,
                dff: int = 2048,
                max_len: int = 100,
                dropout_rate: float = 0.1) -> HandwrittenEquationSolver:
    """
    Create and compile the model.
    
    Args:
        vocab_size: Size of vocabulary
        img_size: Input image size
        d_model: Model dimension
        num_heads: Number of attention heads
        num_decoder_layers: Number of decoder layers
        dff: Feed-forward network dimension
        max_len: Maximum sequence length
        dropout_rate: Dropout rate
    
    Returns:
        Compiled model
    """
    model = HandwrittenEquationSolver(
        vocab_size=vocab_size,
        img_size=img_size,
        d_model=d_model,
        num_heads=num_heads,
        num_decoder_layers=num_decoder_layers,
        dff=dff,
        max_len=max_len,
        dropout_rate=dropout_rate
    )
    
    return model


class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Custom learning rate schedule with warmup."""
    
    def __init__(self, d_model: int, warmup_steps: int = 4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def masked_loss(y_true, y_pred):
    """
    Masked loss function that ignores padding tokens.
    
    Args:
        y_true: True labels (batch, seq_len)
        y_pred: Predicted logits (batch, seq_len, vocab_size)
    
    Returns:
        Loss value
    """
    # Create mask for non-padding tokens (assuming 0 is PAD token)
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    
    # Calculate loss
    loss_object = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )
    loss = loss_object(y_true, y_pred)
    
    # Apply mask
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def masked_accuracy(y_true, y_pred):
    """
    Masked accuracy that ignores padding tokens.
    
    Args:
        y_true: True labels (batch, seq_len)
        y_pred: Predicted logits (batch, seq_len, vocab_size)
    
    Returns:
        Accuracy value
    """
    # Get predictions
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    
    # Create mask
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    
    # Calculate accuracy
    accuracies = tf.equal(y_true, y_pred)
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

