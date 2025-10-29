"""
Script to visualize samples from the dataset.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import CROHME2019Dataset
from src.preprocessing import DataPreprocessor
from src.utils import normalize_strokes, strokes_to_image


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize dataset samples')
    
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory for dataset cache')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to visualize')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--img_size', type=int, default=128,
                       help='Image size for rendering')
    parser.add_argument('--save_path', type=str, default=None,
                       help='Path to save visualization')
    
    return parser.parse_args()


def plot_strokes(strokes, title="Strokes", ax=None):
    """
    Plot raw strokes.
    
    Args:
        strokes: List of stroke arrays
        title: Plot title
        ax: Matplotlib axis (if None, create new)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    for stroke in strokes:
        ax.plot(stroke[:, 0], stroke[:, 1], 'b-', linewidth=2)
    
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert y-axis to match typical coordinate system
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def visualize_samples(dataset, split='train', num_samples=5, img_size=128):
    """
    Visualize samples from dataset.
    
    Args:
        dataset: CROHME2019Dataset instance
        split: Dataset split
        num_samples: Number of samples to show
        img_size: Image size for rendering
    """
    samples = dataset.get_split(split)
    
    if not samples:
        print(f"No samples found in {split} split!")
        return
    
    # Randomly select samples
    indices = np.random.choice(len(samples), min(num_samples, len(samples)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        sample = samples[idx]
        strokes = sample['strokes']
        label = sample['label']
        
        # Plot raw strokes
        plot_strokes(strokes, f"Raw Strokes\nLabel: {label}", ax=axes[i, 0])
        
        # Plot normalized strokes
        normalized_strokes, stats = normalize_strokes(strokes)
        plot_strokes(normalized_strokes, 
                    f"Normalized Strokes\nScale: {stats['scale']:.2f}",
                    ax=axes[i, 1])
        
        # Plot rendered image
        image = strokes_to_image(normalized_strokes, (img_size, img_size))
        axes[i, 2].imshow(image, cmap='gray')
        axes[i, 2].set_title(f"Rendered Image ({img_size}x{img_size})")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    return fig


def print_statistics(dataset):
    """Print detailed dataset statistics."""
    stats = dataset.get_statistics()
    
    print("\n" + "="*80)
    print("Dataset Statistics")
    print("="*80)
    
    for split, split_stats in stats.items():
        print(f"\n{split.upper()} Split:")
        print("-" * 40)
        print(f"  Number of samples:     {split_stats['num_samples']}")
        print(f"  Avg strokes per sample: {split_stats['avg_strokes']:.2f}")
        print(f"  Min strokes:           {split_stats['min_strokes']}")
        print(f"  Max strokes:           {split_stats['max_strokes']}")
        print(f"  Avg label length:      {split_stats['avg_label_length']:.2f}")
        print(f"  Max label length:      {split_stats['max_label_length']}")
    
    # Print label examples
    print("\n" + "="*80)
    print("Sample Labels")
    print("="*80)
    
    all_data = dataset.get_split('train')
    if all_data:
        sample_labels = [sample['label'] for sample in all_data[:10]]
        for i, label in enumerate(sample_labels, 1):
            print(f"{i:2d}. {label}")


def main():
    """Main visualization function."""
    args = parse_args()
    
    print("Loading dataset...")
    dataset = CROHME2019Dataset(cache_dir=args.data_dir)
    dataset.download_and_parse()
    
    # Print statistics
    print_statistics(dataset)
    
    # Visualize samples
    print(f"\nVisualizing {args.num_samples} samples from {args.split} split...")
    fig = visualize_samples(dataset, args.split, args.num_samples, args.img_size)
    
    if fig:
        if args.save_path:
            fig.savefig(args.save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {args.save_path}")
        else:
            plt.show()


if __name__ == '__main__':
    main()

