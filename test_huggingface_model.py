"""
Test loading model from Hugging Face Hub.
Run this after uploading to verify your model works.
"""

import tensorflow as tf
from huggingface_hub import hf_hub_download
import pickle
import numpy as np
import os

def test_model_loading(repo_id):
    """
    Test loading model and vocabulary from Hugging Face.
    
    Args:
        repo_id: Full repository ID (e.g., "username/handwritten-equation-solver")
    """
    print("=" * 80)
    print("TESTING HUGGING FACE MODEL")
    print("=" * 80)
    print(f"\nRepository: {repo_id}")
    
    # Download model
    print("\n1. Downloading model...")
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="best_model.keras"
        )
        print(f"✓ Model downloaded to: {model_path}")
        
        # Check file size
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        return False
    
    # Download vocabulary
    print("\n2. Downloading vocabulary...")
    try:
        vocab_path = hf_hub_download(
            repo_id=repo_id,
            filename="vocabulary.pkl"
        )
        print(f"✓ Vocabulary downloaded to: {vocab_path}")
    except Exception as e:
        print(f"✗ Failed to download vocabulary: {e}")
        return False
    
    # Load model
    print("\n3. Loading model...")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✓ Model loaded successfully!")
        
        # Print model summary
        print("\nModel Architecture:")
        model.summary()
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Load vocabulary
    print("\n4. Loading vocabulary...")
    try:
        with open(vocab_path, 'rb') as f:
            vocabulary = pickle.load(f)
        print(f"✓ Vocabulary loaded successfully!")
        print(f"  Vocabulary size: {len(vocabulary)}")
        print(f"  Special tokens: <SOS>, <EOS>, <PAD>, <UNK>")
    except Exception as e:
        print(f"✗ Failed to load vocabulary: {e}")
        return False
    
    # Test inference
    print("\n5. Testing model inference...")
    try:
        # Create dummy input
        dummy_image = np.random.randn(1, 128, 128, 1).astype(np.float32)
        dummy_decoder_input = np.array([[vocabulary.char2idx['<SOS>']]], dtype=np.int32)
        
        # Run prediction
        predictions = model.predict([dummy_image, dummy_decoder_input], verbose=0)
        print(f"✓ Inference successful!")
        print(f"  Output shape: {predictions.shape}")
        print(f"  Expected: (1, 1, vocab_size)")
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False
    
    # Download config
    print("\n6. Checking configuration...")
    try:
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json"
        )
        print(f"✓ Config downloaded to: {config_path}")
        
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("\nModel Configuration:")
        print(f"  Architecture: {config.get('architecture', {}).get('encoder', 'N/A')} → "
              f"{config.get('architecture', {}).get('decoder', 'N/A')}")
        print(f"  Parameters: d_model={config.get('architecture', {}).get('d_model', 'N/A')}, "
              f"layers={config.get('architecture', {}).get('num_layers', 'N/A')}")
        print(f"  Dataset: {config.get('training', {}).get('dataset', 'N/A')}")
        print(f"  Val Accuracy: {config.get('performance', {}).get('val_accuracy', 'N/A'):.2%}")
    except Exception as e:
        print(f"⚠ Config not found (optional): {e}")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print(f"\nYour model is working correctly on Hugging Face! 🎉")
    print(f"Share it with: https://huggingface.co/{repo_id}")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test model loading from Hugging Face")
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Full repository ID (e.g., 'username/handwritten-equation-solver')"
    )
    
    args = parser.parse_args()
    
    success = test_model_loading(args.repo_id)
    
    if not success:
        print("\n✗ Some tests failed. Check the errors above.")
        exit(1)

