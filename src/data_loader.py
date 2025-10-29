"""
Data loader for CROHME2019 dataset.
Parses InkML files and extracts handwritten strokes and labels.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
from typing import Dict, List, Tuple, Optional
import kagglehub
from tqdm import tqdm
import pickle


class InkMLParser:
    """Parser for InkML files containing handwritten mathematical expressions."""
    
    def __init__(self):
        # Try both common InkML namespaces
        self.namespaces = [
            {'inkml': 'http://www.w3.org/2003/InkML'},  # CROHME2019 uses this
            {'inkml': 'http://www.ink-markup.org/2008/inkml'}
        ]
    
    def parse_file(self, file_path: str) -> Optional[Dict]:
        """
        Parse a single InkML file.
        
        Args:
            file_path: Path to InkML file
        
        Returns:
            Dictionary containing strokes and label, or None if parsing fails
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract strokes
            strokes = self._extract_strokes(root)
            
            # Extract ground truth label (LaTeX format)
            label = self._extract_label(root)
            
            if strokes and label:
                return {
                    'strokes': strokes,
                    'label': label,
                    'file_path': file_path
                }
            
            return None
        
        except Exception as e:
            print(f"Error parsing {file_path}: {str(e)}")
            return None
    
    def _extract_strokes(self, root) -> List[np.ndarray]:
        """Extract stroke data from InkML root."""
        strokes = []
        
        # Find all trace elements - try with different namespaces
        traces = []
        for ns in self.namespaces:
            traces = root.findall('.//inkml:trace', ns)
            if traces:
                break
        
        if not traces:
            # Try without namespace
            traces = root.findall('.//trace')
        
        for trace in traces:
            if trace.text:
                # Parse coordinates
                points = []
                for point_str in trace.text.strip().split(','):
                    coords = point_str.strip().split()
                    if len(coords) >= 2:
                        try:
                            x = float(coords[0])
                            y = float(coords[1])
                            points.append([x, y])
                        except ValueError:
                            continue
                
                if points:
                    strokes.append(np.array(points, dtype=np.float32))
        
        return strokes
    
    def _extract_label(self, root) -> Optional[str]:
        """Extract ground truth label from InkML root."""
        # Try different possible label locations
        
        # Method 1: Look for annotation with type="truth"
        annotations = []
        for ns in self.namespaces:
            annotations = root.findall('.//inkml:annotation[@type="truth"]', ns)
            if annotations:
                break
        
        if not annotations:
            annotations = root.findall('.//annotation[@type="truth"]')
        
        if annotations and annotations[0].text:
            return annotations[0].text.strip()
        
        # Method 2: Look for annotationXML with encoding="LaTeX"
        annotation_xml = []
        for ns in self.namespaces:
            annotation_xml = root.findall('.//inkml:annotationXML[@encoding="LaTeX"]', ns)
            if annotation_xml:
                break
        
        if not annotation_xml:
            annotation_xml = root.findall('.//annotationXML[@encoding="LaTeX"]')
        
        if annotation_xml and annotation_xml[0].text:
            return annotation_xml[0].text.strip()
        
        # Method 3: Look for any annotation with "latex" or "label" in type
        all_annotations = []
        for ns in self.namespaces:
            all_annotations = root.findall('.//inkml:annotation', ns)
            if all_annotations:
                break
        
        if not all_annotations:
            all_annotations = root.findall('.//annotation')
        
        for ann in all_annotations:
            ann_type = ann.get('type', '').lower()
            if 'latex' in ann_type or 'label' in ann_type or 'truth' in ann_type:
                if ann.text:
                    return ann.text.strip()
        
        # Method 4: Look for MathML and convert (simplified)
        mathml = []
        for ns in self.namespaces:
            mathml = root.findall('.//inkml:annotationXML[@encoding="Content-MathML"]', ns)
            if mathml:
                break
        
        if not mathml:
            mathml = root.findall('.//annotationXML[@encoding="Content-MathML"]')
        
        if mathml:
            # For simplicity, extract text content
            text_content = ''.join(mathml[0].itertext()).strip()
            if text_content:
                return text_content
        
        return None


class CROHME2019Dataset:
    """Dataset class for CROHME2019 handwritten math expressions."""
    
    def __init__(self, cache_dir: str = './data'):
        self.cache_dir = cache_dir
        self.parser = InkMLParser()
        self.data = {
            'train': [],
            'val': [],
            'test': []
        }
    
    def download_and_parse(self, force_download: bool = False):
        """
        Download CROHME2019 dataset and parse all InkML files.
        
        Args:
            force_download: If True, re-download even if cache exists
        """
        cache_file = os.path.join(self.cache_dir, 'parsed_data.pkl')
        
        # Check if already parsed
        if os.path.exists(cache_file) and not force_download:
            print("Loading cached dataset...")
            with open(cache_file, 'rb') as f:
                self.data = pickle.load(f)
            print(f"Loaded: Train={len(self.data['train'])}, "
                  f"Val={len(self.data['val'])}, Test={len(self.data['test'])}")
            return
        
        # Download dataset
        print("Downloading CROHME2019 dataset...")
        print("This may take 10-30 minutes depending on your internet speed...")
        try:
            path = kagglehub.dataset_download("ntcuong2103/crohme2019")
            print(f"Dataset downloaded to: {path}")
            
            # List contents to verify download
            print(f"\nDataset contents:")
            for root, dirs, files in os.walk(path):
                level = root.replace(path, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files[:5]:  # Show first 5 files
                    print(f"{subindent}{file}")
                if len(files) > 5:
                    print(f"{subindent}... and {len(files) - 5} more files")
                if level >= 2:  # Don't go too deep
                    break
                    
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            print("Please ensure kagglehub is properly configured with your Kaggle credentials.")
            print("\nTroubleshooting:")
            print("1. Check Kaggle API: ls -la ~/.kaggle/kaggle.json")
            print("2. Test Kaggle API: kaggle datasets list")
            print("3. Check permissions: chmod 600 ~/.kaggle/kaggle.json")
            import traceback
            traceback.print_exc()
            return
        
        # Parse InkML files
        self._parse_dataset(path)
        
        # Cache parsed data
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(self.data, f)
        
        print(f"Dataset parsed and cached: Train={len(self.data['train'])}, "
              f"Val={len(self.data['val'])}, Test={len(self.data['test'])}")
    
    def _parse_dataset(self, dataset_path: str):
        """Parse all InkML files in the dataset."""
        
        # Common directory structures in CROHME datasets
        possible_dirs = [
            ('train', ['train', 'training', 'Train', 'Training']),
            ('val', ['val', 'validation', 'valid', 'Valid', 'Validation', 'Valid']),
            ('test', ['test', 'testing', 'Test', 'Testing'])
        ]
        
        # First, try to find the exact folders
        print(f"\nSearching for InkML files in: {dataset_path}")
        all_inkml_files = []
        for root, dirs, files in os.walk(dataset_path):
            inkml_files = [f for f in files if f.endswith('.inkml')]
            if inkml_files:
                print(f"  Found {len(inkml_files)} InkML files in: {os.path.basename(root)}")
                all_inkml_files.extend([(root, f) for f in inkml_files])
        
        # Walk through dataset directory and map directories to splits
        split_mapping = {}
        for root, dirs, files in os.walk(dataset_path):
            dir_name = os.path.basename(root).lower()
            
            # Check if this directory matches any split
            for split_name, possible_names in possible_dirs:
                for possible_name in possible_names:
                    if dir_name == possible_name.lower():
                        # Count InkML files
                        inkml_count = len([f for f in files if f.endswith('.inkml')])
                        if inkml_count > 0:
                            split_mapping[split_name] = root
                            print(f"\n✓ Found {split_name} split: {root}")
                            print(f"  {inkml_count} InkML files")
                            break
        
        # Parse each split
        if split_mapping:
            for split_name, split_path in split_mapping.items():
                print(f"\nParsing {split_name} split...")
                self._parse_split(split_path, split_name)
            
            # Check if parsing was successful
            total_parsed = sum(len(v) for v in self.data.values())
            print(f"\n✓ Successfully parsed {total_parsed} samples")
            print(f"  Train: {len(self.data['train'])}, Val: {len(self.data['val'])}, Test: {len(self.data['test'])}")
            
            # If we found splits and successfully parsed, we're done!
            if total_parsed > 0:
                return
        
        # If no splits found or parsing failed, parse all files and split them
        if not split_mapping:
            print("\nNo split structure found. Parsing all InkML files...")
            all_data = []
            
            for root, dirs, files in os.walk(dataset_path):
                inkml_files = [f for f in files if f.endswith('.inkml')]
                
                if inkml_files:
                    print(f"Found {len(inkml_files)} InkML files in {root}")
                    
                    for filename in tqdm(inkml_files, desc="Parsing"):
                        file_path = os.path.join(root, filename)
                        parsed = self.parser.parse_file(file_path)
                        if parsed:
                            all_data.append(parsed)
            
            # Split data: 70% train, 15% val, 15% test
            np.random.shuffle(all_data)
            n = len(all_data)
            train_end = int(0.7 * n)
            val_end = int(0.85 * n)
            
            self.data['train'] = all_data[:train_end]
            self.data['val'] = all_data[train_end:val_end]
            self.data['test'] = all_data[val_end:]
    
    def _parse_split(self, directory: str, split_name: str):
        """Parse all InkML files in a directory for a specific split."""
        inkml_files = []
        
        # Recursively find all InkML files
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith('.inkml'):
                    inkml_files.append(os.path.join(root, filename))
        
        print(f"Found {len(inkml_files)} InkML files")
        
        # Parse each file
        for file_path in tqdm(inkml_files, desc=f"Parsing {split_name}"):
            parsed = self.parser.parse_file(file_path)
            if parsed:
                self.data[split_name].append(parsed)
    
    def get_split(self, split: str) -> List[Dict]:
        """Get data for a specific split."""
        return self.data.get(split, [])
    
    def get_all_labels(self) -> List[str]:
        """Get all labels from all splits."""
        labels = []
        for split_data in self.data.values():
            labels.extend([item['label'] for item in split_data])
        return labels
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {}
        
        for split_name, split_data in self.data.items():
            if not split_data:
                continue
            
            num_strokes = [len(item['strokes']) for item in split_data]
            label_lengths = [len(item['label']) for item in split_data]
            
            stats[split_name] = {
                'num_samples': len(split_data),
                'avg_strokes': np.mean(num_strokes),
                'max_strokes': np.max(num_strokes),
                'min_strokes': np.min(num_strokes),
                'avg_label_length': np.mean(label_lengths),
                'max_label_length': np.max(label_lengths)
            }
        
        return stats


if __name__ == '__main__':
    # Test the data loader
    dataset = CROHME2019Dataset(cache_dir='./data')
    dataset.download_and_parse()
    
    # Print statistics
    stats = dataset.get_statistics()
    print("\nDataset Statistics:")
    for split, split_stats in stats.items():
        print(f"\n{split.upper()} Split:")
        for key, value in split_stats.items():
            print(f"  {key}: {value}")
    
    # Show a sample
    train_data = dataset.get_split('train')
    if train_data:
        print(f"\nSample from training set:")
        sample = train_data[0]
        print(f"  Label: {sample['label']}")
        print(f"  Number of strokes: {len(sample['strokes'])}")
        print(f"  First stroke shape: {sample['strokes'][0].shape}")

