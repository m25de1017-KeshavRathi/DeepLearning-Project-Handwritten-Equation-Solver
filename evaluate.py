"""
Evaluation script for handwritten equation solver.
"""

import os
import argparse
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

from src.data_loader import CROHME2019Dataset
from src.preprocessing import DataPreprocessor
from src.model import create_model, masked_loss, masked_accuracy
from src.utils import Vocabulary


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate handwritten equation solver')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to model config (if None, infer from model_path)')
    parser.add_argument('--vocab_path', type=str, default=None,
                       help='Path to vocabulary (if None, use ./models/vocabulary.pkl)')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory for dataset cache')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (None for all)')
    parser.add_argument('--save_predictions', type=str, default=None,
                       help='Path to save predictions')
    
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


def calculate_metrics(predictions, ground_truths, vocabulary):
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: List of predicted sequences
        ground_truths: List of ground truth sequences
        vocabulary: Vocabulary object
    
    Returns:
        Dictionary of metrics
    """
    exact_match = 0
    char_correct = 0
    total_chars = 0
    
    for pred, gt in zip(predictions, ground_truths):
        # Decode sequences
        pred_str = vocabulary.decode(pred, remove_special=True)
        gt_str = vocabulary.decode(gt, remove_special=True)
        
        # Exact match
        if pred_str == gt_str:
            exact_match += 1
        
        # Character-level accuracy
        for p, g in zip(pred_str, gt_str):
            if p == g:
                char_correct += 1
        
        total_chars += max(len(pred_str), len(gt_str))
    
    metrics = {
        'exact_match_accuracy': exact_match / len(predictions) if predictions else 0,
        'character_accuracy': char_correct / total_chars if total_chars > 0 else 0,
        'num_samples': len(predictions)
    }
    
    return metrics


def evaluate(model, preprocessor, samples, batch_size, num_samples=None):
    """
    Evaluate model on samples.
    
    Args:
        model: Trained model
        preprocessor: DataPreprocessor instance
        samples: List of samples to evaluate
        batch_size: Batch size
        num_samples: Number of samples to evaluate
    
    Returns:
        metrics: Dictionary of metrics
        predictions: List of predictions
        ground_truths: List of ground truths
    """
    if num_samples:
        samples = samples[:num_samples]
    
    predictions = []
    ground_truths = []
    
    start_token_idx = preprocessor.vocabulary.char2idx[preprocessor.vocabulary.START_TOKEN]
    end_token_idx = preprocessor.vocabulary.char2idx[preprocessor.vocabulary.END_TOKEN]
    
    print(f"Evaluating {len(samples)} samples...")
    
    for i in tqdm(range(0, len(samples), batch_size)):
        batch_samples = samples[i:i+batch_size]
        
        for sample in batch_samples:
            # Process sample
            image, label, _ = preprocessor.process_sample(
                sample['strokes'],
                sample['label'],
                augment=False
            )
            
            # Expand dimensions for batch
            image = np.expand_dims(image, axis=0)
            
            # Generate prediction
            pred_seq = greedy_decode(
                model, image, start_token_idx, end_token_idx,
                preprocessor.max_label_length
            )
            
            predictions.append(pred_seq)
            ground_truths.append(label)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truths, preprocessor.vocabulary)
    
    return metrics, predictions, ground_truths


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load config
    if args.config_path is None:
        # Try to infer from model path
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
    
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = CROHME2019Dataset(cache_dir=args.data_dir)
    dataset.download_and_parse()
    
    samples = dataset.get_split(args.split)
    print(f"Loaded {len(samples)} samples from {args.split} split")
    
    if not samples:
        print(f"Error: No samples found in {args.split} split!")
        return
    
    # Create model
    print("\nCreating model...")
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
    
    # Evaluate
    print("\n" + "="*80)
    print(f"Evaluating on {args.split} split...")
    print("="*80 + "\n")
    
    metrics, predictions, ground_truths = evaluate(
        model,
        preprocessor,
        samples,
        args.batch_size,
        args.num_samples
    )
    
    # Print results
    print("\n" + "="*80)
    print("Evaluation Results:")
    print("="*80)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # Save predictions if requested
    if args.save_predictions:
        print(f"\nSaving predictions to {args.save_predictions}")
        
        results = []
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            pred_str = vocabulary.decode(pred, remove_special=True)
            gt_str = vocabulary.decode(gt, remove_special=True)
            
            results.append({
                'index': i,
                'prediction': pred_str,
                'ground_truth': gt_str,
                'correct': pred_str == gt_str
            })
        
        with open(args.save_predictions, 'w') as f:
            json.dump({
                'metrics': metrics,
                'predictions': results
            }, f, indent=2)
        
        print(f"Predictions saved!")
    
    # Show some examples
    print("\n" + "="*80)
    print("Sample Predictions:")
    print("="*80)
    
    num_examples = min(10, len(predictions))
    for i in range(num_examples):
        pred_str = vocabulary.decode(predictions[i], remove_special=True)
        gt_str = vocabulary.decode(ground_truths[i], remove_special=True)
        match = "✓" if pred_str == gt_str else "✗"
        
        print(f"\nExample {i+1} {match}")
        print(f"  Ground Truth: {gt_str}")
        print(f"  Prediction:   {pred_str}")


if __name__ == '__main__':
    main()

