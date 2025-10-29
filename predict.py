"""
Prediction script for handwritten equation solver.
Make predictions on individual InkML files or images.
"""

import os
import argparse
import json
import numpy as np
import tensorflow as tf
import cv2

from src.data_loader import InkMLParser
from src.preprocessing import DataPreprocessor
from src.model import create_model
from src.utils import Vocabulary, normalize_strokes, strokes_to_image


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict handwritten equations')
    
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input InkML file or image')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to model config (if None, infer from model_path)')
    parser.add_argument('--vocab_path', type=str, default=None,
                       help='Path to vocabulary (if None, use ./models/vocabulary.pkl)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize input and prediction')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save prediction result')
    
    return parser.parse_args()


def greedy_decode(model, image, start_token_idx, end_token_idx, max_len):
    """
    Greedy decoding for sequence generation.
    
    Args:
        model: Trained model
        image: Input image (1, H, W, 1)
        start_token_idx: Index of start token
        end_token_idx: Index of end token
        max_len: Maximum sequence length
    
    Returns:
        Generated sequence
    """
    # Encode image
    encoder_output = model.encoder(image, training=False)
    
    # Start with start token
    decoder_input = tf.constant([[start_token_idx]], dtype=tf.int32)
    
    # Generate sequence
    for _ in range(max_len - 1):
        # Get predictions
        predictions = model.decoder(decoder_input, encoder_output, training=False)
        
        # Get last prediction
        prediction = predictions[:, -1:, :]
        predicted_id = tf.argmax(prediction, axis=-1, output_type=tf.int32)
        
        # Append to decoder input
        decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)
        
        # Stop if end token is generated
        if predicted_id == end_token_idx:
            break
    
    return decoder_input.numpy()[0]


def beam_search_decode(model, image, start_token_idx, end_token_idx, max_len, beam_width=5):
    """
    Beam search decoding for sequence generation.
    
    Args:
        model: Trained model
        image: Input image (1, H, W, 1)
        start_token_idx: Index of start token
        end_token_idx: Index of end token
        max_len: Maximum sequence length
        beam_width: Beam width
    
    Returns:
        Best sequence
    """
    # Encode image
    encoder_output = model.encoder(image, training=False)
    
    # Initialize beams
    beams = [([start_token_idx], 0.0)]  # (sequence, log_prob)
    
    for _ in range(max_len - 1):
        candidates = []
        
        for sequence, score in beams:
            # Check if sequence ended
            if sequence[-1] == end_token_idx:
                candidates.append((sequence, score))
                continue
            
            # Prepare decoder input
            decoder_input = tf.constant([sequence], dtype=tf.int32)
            
            # Get predictions
            predictions = model.decoder(decoder_input, encoder_output, training=False)
            
            # Get last prediction
            prediction = predictions[0, -1, :]
            log_probs = tf.nn.log_softmax(prediction).numpy()
            
            # Get top k tokens
            top_k = np.argsort(log_probs)[-beam_width:]
            
            for token_id in top_k:
                new_sequence = sequence + [int(token_id)]
                new_score = score + log_probs[token_id]
                candidates.append((new_sequence, new_score))
        
        # Select top beams
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]
        
        # Check if all beams ended
        if all(seq[-1] == end_token_idx for seq, _ in beams):
            break
    
    # Return best sequence
    best_sequence, _ = beams[0]
    return np.array(best_sequence)


def load_image(image_path, img_size):
    """
    Load and preprocess image.
    
    Args:
        image_path: Path to image file
        img_size: Target image size
    
    Returns:
        Preprocessed image
    """
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Resize
    image = cv2.resize(image, (img_size, img_size))
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Add channel dimension
    image = np.expand_dims(image, axis=-1)
    
    return image


def predict_inkml(inkml_path, model, preprocessor, use_beam_search=False, beam_width=5):
    """
    Make prediction on InkML file.
    
    Args:
        inkml_path: Path to InkML file
        model: Trained model
        preprocessor: DataPreprocessor instance
        use_beam_search: Whether to use beam search
        beam_width: Beam width for beam search
    
    Returns:
        prediction: Predicted LaTeX string
        image: Rendered image
    """
    # Parse InkML
    parser = InkMLParser()
    parsed = parser.parse_file(inkml_path)
    
    if parsed is None:
        raise ValueError(f"Could not parse InkML file: {inkml_path}")
    
    strokes = parsed['strokes']
    
    # Normalize strokes
    normalized_strokes, _ = normalize_strokes(strokes)
    
    # Convert to image
    image = strokes_to_image(normalized_strokes, 
                            (preprocessor.img_size[0], preprocessor.img_size[1]))
    
    # Normalize image
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)
    
    # Expand batch dimension
    image_batch = np.expand_dims(image, axis=0)
    
    # Get token indices
    start_token_idx = preprocessor.vocabulary.char2idx[preprocessor.vocabulary.START_TOKEN]
    end_token_idx = preprocessor.vocabulary.char2idx[preprocessor.vocabulary.END_TOKEN]
    
    # Generate prediction
    if use_beam_search:
        pred_seq = beam_search_decode(
            model, image_batch, start_token_idx, end_token_idx,
            preprocessor.max_label_length, beam_width
        )
    else:
        pred_seq = greedy_decode(
            model, image_batch, start_token_idx, end_token_idx,
            preprocessor.max_label_length
        )
    
    # Decode prediction
    prediction = preprocessor.vocabulary.decode(pred_seq, remove_special=True)
    
    return prediction, image


def predict_image(image_path, model, preprocessor, use_beam_search=False, beam_width=5):
    """
    Make prediction on image file.
    
    Args:
        image_path: Path to image file
        model: Trained model
        preprocessor: DataPreprocessor instance
        use_beam_search: Whether to use beam search
        beam_width: Beam width for beam search
    
    Returns:
        prediction: Predicted LaTeX string
        image: Loaded image
    """
    # Load image
    image = load_image(image_path, preprocessor.img_size[0])
    
    # Expand batch dimension
    image_batch = np.expand_dims(image, axis=0)
    
    # Get token indices
    start_token_idx = preprocessor.vocabulary.char2idx[preprocessor.vocabulary.START_TOKEN]
    end_token_idx = preprocessor.vocabulary.char2idx[preprocessor.vocabulary.END_TOKEN]
    
    # Generate prediction
    if use_beam_search:
        pred_seq = beam_search_decode(
            model, image_batch, start_token_idx, end_token_idx,
            preprocessor.max_label_length, beam_width
        )
    else:
        pred_seq = greedy_decode(
            model, image_batch, start_token_idx, end_token_idx,
            preprocessor.max_label_length
        )
    
    # Decode prediction
    prediction = preprocessor.vocabulary.decode(pred_seq, remove_special=True)
    
    return prediction, image


def visualize_prediction(image, prediction, ground_truth=None):
    """
    Visualize prediction.
    
    Args:
        image: Input image
        prediction: Predicted LaTeX string
        ground_truth: Ground truth LaTeX string (optional)
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 4))
    
    # Show image
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image.squeeze()
    
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    title = f"Prediction: {prediction}"
    if ground_truth:
        title = f"Ground Truth: {ground_truth}\n{title}"
    
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def main():
    """Main prediction function."""
    args = parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Load config
    if args.config_path is None:
        model_dir = os.path.dirname(args.model_path)
        config_files = [f for f in os.listdir(model_dir) if f.startswith('config_') and f.endswith('.json')]
        if config_files:
            args.config_path = os.path.join(model_dir, config_files[-1])
        else:
            args.config_path = os.path.join(model_dir, 'config.json')
    
    if os.path.exists(args.config_path):
        print(f"Loading config from {args.config_path}")
        with open(args.config_path, 'r') as f:
            config = json.load(f)
    else:
        print("Config file not found, using default parameters")
        config = {
            'img_size': 128,
            'd_model': 512,
            'num_heads': 8,
            'num_layers': 4,
            'dff': 2048,
            'dropout': 0.1,
            'max_len': 100
        }
    
    # Load vocabulary
    if args.vocab_path is None:
        args.vocab_path = './models/vocabulary.pkl'
    
    print(f"Loading vocabulary from {args.vocab_path}")
    vocabulary = Vocabulary()
    vocabulary.load(args.vocab_path)
    vocab_size = len(vocabulary)
    print(f"Vocabulary size: {vocab_size}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        img_size=(config['img_size'], config['img_size']),
        max_label_length=config['max_len']
    )
    preprocessor.vocabulary = vocabulary
    
    # Create model
    print("Creating model...")
    model = create_model(
        vocab_size=vocab_size,
        img_size=(config['img_size'], config['img_size']),
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_decoder_layers=config['num_layers'],
        dff=config['dff'],
        max_len=config['max_len'],
        dropout_rate=config['dropout']
    )
    
    # Build model
    dummy_input = {
        'input_image': tf.zeros((1, config['img_size'], config['img_size'], 1)),
        'input_label': tf.zeros((1, config['max_len'] - 1), dtype=tf.int32)
    }
    _ = model(dummy_input)
    
    # Load weights
    print(f"Loading model weights from {args.model_path}")
    model.load_weights(args.model_path)
    
    # Determine input type
    input_ext = os.path.splitext(args.input)[1].lower()
    
    print(f"\nMaking prediction on {args.input}...")
    
    try:
        if input_ext == '.inkml':
            prediction, image = predict_inkml(args.input, model, preprocessor)
        else:
            # Assume it's an image
            prediction, image = predict_image(args.input, model, preprocessor)
        
        # Print prediction
        print("\n" + "="*80)
        print("Prediction Result:")
        print("="*80)
        print(f"LaTeX: {prediction}")
        print("="*80)
        
        # Save result if requested
        if args.output:
            result = {
                'input_file': args.input,
                'prediction': prediction
            }
            
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"\nResult saved to {args.output}")
        
        # Visualize if requested
        if args.visualize:
            visualize_prediction(image, prediction)
    
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

