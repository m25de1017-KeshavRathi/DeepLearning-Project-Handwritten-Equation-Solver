"""
Training script for handwritten equation solver.
"""

import os
import argparse
import json
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

from src.data_loader import CROHME2019Dataset
from src.preprocessing import DataPreprocessor, create_data_generators
from src.model import create_model, CustomSchedule, masked_loss, masked_accuracy
from src.utils import create_directories


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train handwritten equation solver')
    
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory for dataset cache')
    parser.add_argument('--model_dir', type=str, default='./models',
                       help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for logs')
    
    # Model hyperparameters
    parser.add_argument('--img_size', type=int, default=128,
                       help='Image size (will be img_size x img_size)')
    parser.add_argument('--d_model', type=int, default=512,
                       help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of decoder layers')
    parser.add_argument('--dff', type=int, default=2048,
                       help='Feed-forward network dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--max_len', type=int, default=100,
                       help='Maximum sequence length')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (if None, use custom schedule)')
    parser.add_argument('--warmup_steps', type=int, default=4000,
                       help='Warmup steps for learning rate schedule')
    parser.add_argument('--augment', action='store_true',
                       help='Use data augmentation')
    parser.add_argument('--min_freq', type=int, default=1,
                       help='Minimum frequency for vocabulary')
    
    # Other options
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_freq', type=int, default=5,
                       help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--force_download', action='store_true',
                       help='Force re-download and parse dataset')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Create directories
    create_directories('.')
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set up GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU found, using CPU")
    
    # Load dataset
    print("Loading dataset...")
    dataset = CROHME2019Dataset(cache_dir=args.data_dir)
    dataset.download_and_parse(force_download=args.force_download)
    
    train_data = dataset.get_split('train')
    val_data = dataset.get_split('val')
    
    if not train_data:
        print("Error: No training data found!")
        return
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Print dataset statistics
    stats = dataset.get_statistics()
    print("\nDataset Statistics:")
    for split, split_stats in stats.items():
        print(f"\n{split.upper()} Split:")
        for key, value in split_stats.items():
            print(f"  {key}: {value}")
    
    # Initialize preprocessor and build vocabulary
    print("\nBuilding vocabulary...")
    preprocessor = DataPreprocessor(
        img_size=(args.img_size, args.img_size),
        max_label_length=args.max_len
    )
    
    all_labels = dataset.get_all_labels()
    preprocessor.build_vocabulary(all_labels, min_freq=args.min_freq)
    
    vocab_size = len(preprocessor.vocabulary)
    print(f"Vocabulary size: {vocab_size}")
    
    # Save vocabulary
    vocab_path = os.path.join(args.model_dir, 'vocabulary.pkl')
    preprocessor.vocabulary.save(vocab_path)
    print(f"Vocabulary saved to {vocab_path}")
    
    # Create data generators
    print("\nCreating data generators...")
    train_gen, val_gen = create_data_generators(
        train_data,
        val_data,
        preprocessor,
        batch_size=args.batch_size,
        augment_train=args.augment
    )
    
    print(f"Training batches per epoch: {len(train_gen)}")
    print(f"Validation batches per epoch: {len(val_gen)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        vocab_size=vocab_size,
        img_size=(args.img_size, args.img_size),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_decoder_layers=args.num_layers,
        dff=args.dff,
        max_len=args.max_len,
        dropout_rate=args.dropout
    )
    
    # Set up learning rate schedule
    if args.learning_rate is None:
        learning_rate = CustomSchedule(args.d_model, warmup_steps=args.warmup_steps)
        print(f"Using custom learning rate schedule with warmup_steps={args.warmup_steps}")
    else:
        learning_rate = args.learning_rate
        print(f"Using fixed learning rate: {learning_rate}")
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    model.compile(
        optimizer=optimizer,
        loss=masked_loss,
        metrics=[masked_accuracy]
    )
    
    # Build model
    dummy_input = {
        'input_image': tf.zeros((1, args.img_size, args.img_size, 1)),
        'input_label': tf.zeros((1, args.max_len - 1), dtype=tf.int32)
    }
    _ = model(dummy_input)
    
    print("\nModel Summary:")
    model.summary()
    
    # Calculate total parameters
    total_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"\nTotal trainable parameters: {total_params:,}")
    
    # Set up callbacks
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{timestamp}"
    
    callbacks = [
        # Save best model based on validation loss
        keras.callbacks.ModelCheckpoint(
            os.path.join(args.model_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        # Save checkpoint every epoch that improves
        keras.callbacks.ModelCheckpoint(
            os.path.join(args.model_dir, f'model_{run_name}_epoch_{{epoch:02d}}_val_loss_{{val_loss:.4f}}.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        # Save checkpoint every N epochs (regardless of improvement)
        keras.callbacks.ModelCheckpoint(
            os.path.join(args.model_dir, f'checkpoint_{run_name}_epoch_{{epoch:02d}}.h5'),
            save_freq=args.checkpoint_freq,  # Save every N epochs
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.log_dir, run_name),
            histogram_freq=1,
            write_graph=True
        ),
        keras.callbacks.CSVLogger(
            os.path.join(args.log_dir, f'training_{run_name}.csv')
        ),
        # Note: ReduceLROnPlateau is not compatible with LearningRateSchedule
        # The custom schedule already handles learning rate adjustments
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Save config
    config = {
        'img_size': args.img_size,
        'd_model': args.d_model,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'dff': args.dff,
        'dropout': args.dropout,
        'max_len': args.max_len,
        'vocab_size': vocab_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'warmup_steps': args.warmup_steps,
        'augment': args.augment,
        'min_freq': args.min_freq,
        'timestamp': timestamp
    }
    
    config_path = os.path.join(args.model_dir, f'config_{run_name}.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfiguration saved to {config_path}")
    
    # Resume from checkpoint if specified
    initial_epoch = 0
    if args.resume:
        print(f"\nLoading checkpoint from {args.resume}...")
        model.load_weights(args.resume)
        # Try to extract epoch from filename
        try:
            initial_epoch = int(args.resume.split('epoch_')[1].split('_')[0])
            print(f"Resuming from epoch {initial_epoch}")
        except:
            print("Could not determine epoch from filename, starting from 0")
    
    # Train model
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, f'final_model_{run_name}.h5')
    model.save_weights(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(args.log_dir, f'history_{run_name}.json')
    with open(history_path, 'w') as f:
        # Convert numpy values to Python types
        history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)


if __name__ == '__main__':
    main()

