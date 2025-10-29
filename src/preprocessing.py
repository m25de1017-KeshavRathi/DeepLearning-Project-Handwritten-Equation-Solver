"""
Data preprocessing and augmentation for handwritten equation recognition.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict
import tensorflow as tf
from src.utils import normalize_strokes, strokes_to_image, Vocabulary


class DataPreprocessor:
    """Preprocessor for converting strokes to model-ready format."""
    
    def __init__(self, 
                 img_size: Tuple[int, int] = (128, 128),
                 max_label_length: int = 100):
        """
        Initialize preprocessor.
        
        Args:
            img_size: Target image size (height, width)
            max_label_length: Maximum label sequence length
        """
        self.img_size = img_size
        self.max_label_length = max_label_length
        self.vocabulary = Vocabulary()
    
    def build_vocabulary(self, labels: List[str], min_freq: int = 1):
        """Build vocabulary from labels."""
        self.vocabulary.build_vocab(labels, min_freq)
        print(f"Vocabulary size: {len(self.vocabulary)}")
    
    def process_sample(self, 
                       strokes: List[np.ndarray], 
                       label: str,
                       augment: bool = False) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Process a single sample.
        
        Args:
            strokes: List of stroke arrays
            label: Ground truth label
            augment: Whether to apply augmentation
        
        Returns:
            image: Processed image
            label_encoded: Encoded label
            label_length: Actual label length
        """
        # Normalize strokes
        normalized_strokes, _ = normalize_strokes(strokes)
        
        # Apply augmentation if requested
        if augment:
            normalized_strokes = self.augment_strokes(normalized_strokes)
        
        # Convert to image
        image = strokes_to_image(normalized_strokes, self.img_size)
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add channel dimension
        image = np.expand_dims(image, axis=-1)
        
        # Encode label
        label_encoded = self.vocabulary.encode(label, add_start_end=True)
        label_length = len(label_encoded)
        
        # Pad label
        if len(label_encoded) > self.max_label_length:
            label_encoded = label_encoded[:self.max_label_length]
            label_length = self.max_label_length
        else:
            label_encoded = label_encoded + [self.vocabulary.char2idx[self.vocabulary.PAD_TOKEN]] * \
                           (self.max_label_length - len(label_encoded))
        
        return image, np.array(label_encoded), label_length
    
    def augment_strokes(self, strokes: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply data augmentation to strokes.
        
        Args:
            strokes: List of normalized stroke arrays
        
        Returns:
            Augmented strokes
        """
        augmented = []
        
        # Random rotation (-15 to 15 degrees)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            angle_rad = np.deg2rad(angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            # Rotate around center (0.5, 0.5) since strokes are normalized
            center = np.array([0.5, 0.5])
            for stroke in strokes:
                centered = stroke - center
                rotated = centered @ rotation_matrix.T
                augmented.append(rotated + center)
        else:
            augmented = [s.copy() for s in strokes]
        
        # Random scaling (0.9 to 1.1)
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            center = np.array([0.5, 0.5])
            augmented = [(s - center) * scale + center for s in augmented]
        
        # Random translation
        if np.random.random() > 0.5:
            tx = np.random.uniform(-0.1, 0.1)
            ty = np.random.uniform(-0.1, 0.1)
            augmented = [s + np.array([tx, ty]) for s in augmented]
        
        # Clip to valid range
        augmented = [np.clip(s, 0, 1) for s in augmented]
        
        return augmented
    
    def process_batch(self, 
                      samples: List[Dict],
                      augment: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a batch of samples.
        
        Args:
            samples: List of dictionaries with 'strokes' and 'label'
            augment: Whether to apply augmentation
        
        Returns:
            images: Batch of images
            labels: Batch of encoded labels
            label_lengths: Batch of label lengths
        """
        images = []
        labels = []
        label_lengths = []
        
        for sample in samples:
            img, lbl, lbl_len = self.process_sample(
                sample['strokes'], 
                sample['label'],
                augment=augment
            )
            images.append(img)
            labels.append(lbl)
            label_lengths.append(lbl_len)
        
        return (np.array(images), 
                np.array(labels), 
                np.array(label_lengths))


class DataGenerator(tf.keras.utils.Sequence):
    """Data generator for training."""
    
    def __init__(self,
                 samples: List[Dict],
                 preprocessor: DataPreprocessor,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 augment: bool = False):
        """
        Initialize data generator.
        
        Args:
            samples: List of samples
            preprocessor: DataPreprocessor instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            augment: Whether to apply augmentation
        """
        self.samples = samples
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.samples))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(len(self.samples) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data."""
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.samples))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Get batch samples
        batch_samples = [self.samples[i] for i in batch_indices]
        
        # Process batch
        images, labels, label_lengths = self.preprocessor.process_batch(
            batch_samples, 
            augment=self.augment
        )
        
        # Ensure correct dtypes
        images = images.astype(np.float32)
        labels = labels.astype(np.int32)
        
        return {'input_image': images, 'input_label': labels[:, :-1]}, \
               labels[:, 1:]
    
    def on_epoch_end(self):
        """Shuffle data after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_data_generators(train_data: List[Dict],
                           val_data: List[Dict],
                           preprocessor: DataPreprocessor,
                           batch_size: int = 32,
                           augment_train: bool = True):
    """
    Create training and validation data generators.
    
    Args:
        train_data: Training samples
        val_data: Validation samples
        preprocessor: DataPreprocessor instance
        batch_size: Batch size
        augment_train: Whether to augment training data
    
    Returns:
        train_generator: Training data generator
        val_generator: Validation data generator
    """
    train_gen = DataGenerator(
        train_data,
        preprocessor,
        batch_size=batch_size,
        shuffle=True,
        augment=augment_train
    )
    
    val_gen = DataGenerator(
        val_data,
        preprocessor,
        batch_size=batch_size,
        shuffle=False,
        augment=False
    )
    
    return train_gen, val_gen


def visualize_sample(image: np.ndarray, label: str, prediction: str = None):
    """
    Visualize a sample with its label and optional prediction.
    
    Args:
        image: Image array
        label: Ground truth label
        prediction: Predicted label (optional)
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 4))
    
    # Show image
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image.squeeze()
    
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    title = f"Label: {label}"
    if prediction:
        title += f"\nPrediction: {prediction}"
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

