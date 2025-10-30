"""
Quick training script for testing (smaller model, fewer epochs).
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from train import *

if __name__ == '__main__':
    import sys
    
    # Override default arguments for quick training
    sys.argv.extend([
        '--epochs', '10',
        '--batch_size', '16', 
        '--d_model', '256',
        '--num_layers', '2',
        '--num_heads', '4',
        '--dff', '1024',
        '--augment',  # CRITICAL: Prevents overfitting!
        '--checkpoint_freq', '5'
    ])
    
    print("="*80)
    print("QUICK TRAINING MODE - Small model for testing")
    print("="*80)
    print("Configuration:")
    print("  - Epochs: 10 (instead of 100)")
    print("  - Batch size: 16 (instead of 32)")
    print("  - Model size: 256 (instead of 512) - ~1.5M parameters")
    print("  - Layers: 2 (instead of 4)")
    print("  - Data augmentation: ENABLED")
    print("  - Time estimate: ~30-60 minutes")
    print("="*80)
    print()
    
    main()

