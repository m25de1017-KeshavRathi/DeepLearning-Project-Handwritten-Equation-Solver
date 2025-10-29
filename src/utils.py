"""
Utility functions for the handwritten equation solver.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple
import pickle


class Vocabulary:
    """Vocabulary class for managing symbol tokens."""
    
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.char_count = {}
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        self.UNK_TOKEN = '<UNK>'
        
        # Initialize with special tokens
        self.add_char(self.PAD_TOKEN)
        self.add_char(self.START_TOKEN)
        self.add_char(self.END_TOKEN)
        self.add_char(self.UNK_TOKEN)
    
    def add_char(self, char: str):
        """Add a character to the vocabulary."""
        if char not in self.char2idx:
            idx = len(self.char2idx)
            self.char2idx[char] = idx
            self.idx2char[idx] = char
            self.char_count[char] = 1
        else:
            self.char_count[char] += 1
    
    def build_vocab(self, labels: List[str], min_freq: int = 1):
        """Build vocabulary from a list of labels."""
        # Count all characters
        for label in labels:
            for char in label:
                self.add_char(char)
        
        # Filter by minimum frequency if needed
        if min_freq > 1:
            filtered_chars = {char for char, count in self.char_count.items() 
                            if count >= min_freq or char in [self.PAD_TOKEN, self.START_TOKEN, 
                                                             self.END_TOKEN, self.UNK_TOKEN]}
            
            # Rebuild vocabulary
            self.char2idx = {}
            self.idx2char = {}
            for char in filtered_chars:
                idx = len(self.char2idx)
                self.char2idx[char] = idx
                self.idx2char[idx] = char
    
    def encode(self, text: str, add_start_end: bool = True) -> List[int]:
        """Convert text to sequence of indices."""
        indices = []
        if add_start_end:
            indices.append(self.char2idx[self.START_TOKEN])
        
        for char in text:
            if char in self.char2idx:
                indices.append(self.char2idx[char])
            else:
                indices.append(self.char2idx[self.UNK_TOKEN])
        
        if add_start_end:
            indices.append(self.char2idx[self.END_TOKEN])
        
        return indices
    
    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """Convert sequence of indices to text."""
        chars = []
        special_tokens = {self.char2idx[self.PAD_TOKEN], 
                         self.char2idx[self.START_TOKEN], 
                         self.char2idx[self.END_TOKEN]}
        
        for idx in indices:
            if remove_special and idx in special_tokens:
                continue
            if idx in self.idx2char:
                chars.append(self.idx2char[idx])
        
        return ''.join(chars)
    
    def __len__(self):
        return len(self.char2idx)
    
    def save(self, path: str):
        """Save vocabulary to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'char2idx': self.char2idx,
                'idx2char': self.idx2char,
                'char_count': self.char_count
            }, f)
    
    def load(self, path: str):
        """Load vocabulary from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.char2idx = data['char2idx']
            self.idx2char = data['idx2char']
            self.char_count = data['char_count']


def normalize_strokes(strokes: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict]:
    """
    Normalize stroke coordinates.
    
    Args:
        strokes: List of numpy arrays of shape (n_points, 2)
    
    Returns:
        normalized_strokes: List of normalized stroke arrays
        stats: Dictionary containing normalization statistics
    """
    if not strokes:
        return [], {}
    
    # Concatenate all points
    all_points = np.vstack(strokes)
    
    # Calculate statistics
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)
    
    width = x_max - x_min
    height = y_max - y_min
    
    # Avoid division by zero
    if width == 0:
        width = 1
    if height == 0:
        height = 1
    
    scale = max(width, height)
    
    # Normalize to [0, 1] range
    normalized_strokes = []
    for stroke in strokes:
        normalized = (stroke - np.array([x_min, y_min])) / scale
        normalized_strokes.append(normalized)
    
    stats = {
        'x_min': float(x_min),
        'y_min': float(y_min),
        'scale': float(scale),
        'width': float(width),
        'height': float(height)
    }
    
    return normalized_strokes, stats


def strokes_to_image(strokes: List[np.ndarray], 
                     img_size: Tuple[int, int] = (128, 128),
                     line_width: int = 2) -> np.ndarray:
    """
    Convert strokes to image representation.
    
    Args:
        strokes: List of normalized stroke arrays
        img_size: Target image size (height, width)
        line_width: Line thickness
    
    Returns:
        image: Binary image array
    """
    import cv2
    
    height, width = img_size
    image = np.zeros((height, width), dtype=np.uint8)
    
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        
        # Scale to image size
        scaled_stroke = stroke * np.array([width - 1, height - 1])
        scaled_stroke = scaled_stroke.astype(np.int32)
        
        # Draw lines
        for i in range(len(scaled_stroke) - 1):
            pt1 = tuple(scaled_stroke[i])
            pt2 = tuple(scaled_stroke[i + 1])
            cv2.line(image, pt1, pt2, 255, line_width)
    
    return image


def pad_sequences(sequences: List[np.ndarray], 
                  maxlen: int = None,
                  padding: str = 'post',
                  value: float = 0.0) -> np.ndarray:
    """
    Pad sequences to the same length.
    
    Args:
        sequences: List of sequences
        maxlen: Maximum length (if None, use longest sequence)
        padding: 'pre' or 'post'
        value: Padding value
    
    Returns:
        Padded sequences as numpy array
    """
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    
    padded = np.full((len(sequences), maxlen, *sequences[0].shape[1:]), 
                     value, dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        length = min(len(seq), maxlen)
        if padding == 'post':
            padded[i, :length] = seq[:length]
        else:  # 'pre'
            padded[i, -length:] = seq[:length]
    
    return padded


def create_directories(base_path: str):
    """Create necessary directories for the project."""
    directories = [
        'models',
        'logs',
        'data',
        'data/processed',
        'data/raw'
    ]
    
    for directory in directories:
        path = os.path.join(base_path, directory)
        os.makedirs(path, exist_ok=True)


def save_config(config: Dict, path: str):
    """Save configuration to JSON file."""
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

