"""
Example usage script demonstrating the complete workflow.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

from src.data_loader import CROHME2019Dataset
from src.preprocessing import DataPreprocessor, create_data_generators
from src.model import create_model, CustomSchedule, masked_loss, masked_accuracy
from src.utils import create_directories
import tensorflow as tf


def main():
    """Example workflow for training and using the model."""
    
    print("="*80)
    print("Handwritten Equation Solver - Example Usage")
    print("="*80)
    
    # Step 1: Download and parse dataset
    print("\nStep 1: Loading dataset...")
    print("-" * 80)
    
    dataset = CROHME2019Dataset(cache_dir='./data')
    dataset.download_and_parse(force_download=False)
    
    train_data = dataset.get_split('train')
    val_data = dataset.get_split('val')
    test_data = dataset.get_split('test')
    
    print(f"✓ Training samples: {len(train_data)}")
    print(f"✓ Validation samples: {len(val_data)}")
    print(f"✓ Test samples: {len(test_data)}")
    
    # Check if dataset was loaded
    if not train_data:
        print("\n❌ Error: No training data found!")
        print("The dataset download may have failed or the dataset structure is unexpected.")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify Kaggle API credentials: ls -la ~/.kaggle/kaggle.json")
        print("3. Try manual download: python src/data_loader.py")
        print("4. Check if data/parsed_data.pkl exists")
        return
    
    # Print statistics
    stats = dataset.get_statistics()
    print("\nDataset Statistics:")
    for split, split_stats in stats.items():
        print(f"  {split}: {split_stats['num_samples']} samples")
    
    # Step 2: Build vocabulary
    print("\nStep 2: Building vocabulary...")
    print("-" * 80)
    
    preprocessor = DataPreprocessor(img_size=(128, 128), max_label_length=100)
    all_labels = dataset.get_all_labels()
    preprocessor.build_vocabulary(all_labels, min_freq=1)
    
    vocab_size = len(preprocessor.vocabulary)
    print(f"✓ Vocabulary size: {vocab_size}")
    
    # Show sample characters
    sample_chars = list(preprocessor.vocabulary.char2idx.keys())[:20]
    print(f"✓ Sample characters: {', '.join(sample_chars)}")
    
    # Step 3: Create data generators
    print("\nStep 3: Creating data generators...")
    print("-" * 80)
    
    batch_size = 8  # Small batch for demo
    train_gen, val_gen = create_data_generators(
        train_data[:100],  # Use small subset for demo
        val_data[:20],
        preprocessor,
        batch_size=batch_size,
        augment_train=True
    )
    
    print(f"✓ Training batches: {len(train_gen)}")
    print(f"✓ Validation batches: {len(val_gen)}")
    
    # Step 4: Create model
    print("\nStep 4: Creating model...")
    print("-" * 80)
    
    model = create_model(
        vocab_size=vocab_size,
        img_size=(128, 128),
        d_model=256,  # Smaller for demo
        num_heads=4,
        num_decoder_layers=2,
        dff=1024,
        max_len=100,
        dropout_rate=0.1
    )
    
    # Build model
    dummy_input = {
        'input_image': tf.zeros((1, 128, 128, 1)),
        'input_label': tf.zeros((1, 99), dtype=tf.int32)
    }
    _ = model(dummy_input)
    
    total_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"✓ Model created")
    print(f"✓ Total parameters: {total_params:,}")
    
    # Step 5: Compile model
    print("\nStep 5: Compiling model...")
    print("-" * 80)
    
    learning_rate = CustomSchedule(256, warmup_steps=1000)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    model.compile(
        optimizer=optimizer,
        loss=masked_loss,
        metrics=[masked_accuracy]
    )
    
    print("✓ Model compiled")
    
    # Step 6: Train model (just a few steps for demo)
    print("\nStep 6: Training model (demo - 2 epochs)...")
    print("-" * 80)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=2,
        verbose=1
    )
    
    print("\n✓ Training completed")
    
    # Step 7: Show example prediction
    print("\nStep 7: Making example prediction...")
    print("-" * 80)
    
    # Get a sample
    sample = train_data[0]
    image, label, _ = preprocessor.process_sample(
        sample['strokes'],
        sample['label'],
        augment=False
    )
    
    # Make prediction
    image_batch = tf.expand_dims(image, axis=0)
    encoder_output = model.encoder(image_batch, training=False)
    
    start_token = preprocessor.vocabulary.char2idx[preprocessor.vocabulary.START_TOKEN]
    end_token = preprocessor.vocabulary.char2idx[preprocessor.vocabulary.END_TOKEN]
    
    decoder_input = tf.constant([[start_token]], dtype=tf.int32)
    
    # Simple greedy decoding
    for _ in range(20):
        predictions = model.decoder(decoder_input, encoder_output, training=False)
        prediction = predictions[:, -1:, :]
        predicted_id = tf.argmax(prediction, axis=-1, output_type=tf.int32)
        decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)
        
        if predicted_id == end_token:
            break
    
    pred_seq = decoder_input.numpy()[0]
    pred_str = preprocessor.vocabulary.decode(pred_seq, remove_special=True)
    
    print(f"Ground Truth: {sample['label']}")
    print(f"Prediction:   {pred_str}")
    
    # Step 8: Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print("✓ Dataset loaded and parsed")
    print("✓ Vocabulary built")
    print("✓ Model created and compiled")
    print("✓ Model trained (demo)")
    print("✓ Prediction made")
    print("\nNext steps:")
    print("  1. Train full model: python train.py --epochs 100 --batch_size 32")
    print("  2. Evaluate: python evaluate.py --model_path models/best_model.h5")
    print("  3. Predict: python predict.py --input path/to/file.inkml --model_path models/best_model.h5")
    print("="*80)


if __name__ == '__main__':
    main()

