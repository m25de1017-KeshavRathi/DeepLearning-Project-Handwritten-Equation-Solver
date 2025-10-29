"""
Test script to verify the installation and setup.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    packages = {
        'numpy': 'NumPy',
        'tensorflow': 'TensorFlow',
        'cv2': 'OpenCV',
        'matplotlib': 'Matplotlib',
        'sklearn': 'scikit-learn',
        'pandas': 'Pandas',
        'PIL': 'Pillow',
        'kagglehub': 'KaggleHub',
        'lxml': 'lxml',
        'tqdm': 'tqdm'
    }
    
    failed = []
    
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} - {str(e)}")
            failed.append(name)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True


def test_project_structure():
    """Test if project structure is correct."""
    print("\nTesting project structure...")
    
    required_files = [
        'train.py',
        'evaluate.py',
        'predict.py',
        'example_usage.py',
        'visualize_data.py',
        'requirements.txt',
        'README.md',
        'src/data_loader.py',
        'src/preprocessing.py',
        'src/model.py',
        'src/utils.py'
    ]
    
    required_dirs = [
        'src',
        'data',
        'models',
        'logs'
    ]
    
    failed = []
    
    # Check files
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} - Not found")
            failed.append(filepath)
    
    # Check directories
    for dirpath in required_dirs:
        if os.path.isdir(dirpath):
            print(f"  ✓ {dirpath}/")
        else:
            print(f"  ✗ {dirpath}/ - Not found")
            failed.append(dirpath)
    
    if failed:
        print(f"\n❌ Missing: {', '.join(failed)}")
        return False
    else:
        print("\n✅ Project structure is correct!")
        return True


def test_tensorflow():
    """Test TensorFlow installation and GPU availability."""
    print("\nTesting TensorFlow...")
    
    try:
        import tensorflow as tf
        
        print(f"  TensorFlow version: {tf.__version__}")
        
        # Check GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  ✓ Found {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"    - {gpu.name}")
        else:
            print("  ⚠ No GPU found - training will be slow on CPU")
        
        # Test basic operation
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = tf.constant([[1.0], [1.0]])
        z = tf.matmul(x, y)
        print(f"  ✓ Basic TensorFlow operation works")
        
        return True
    
    except Exception as e:
        print(f"  ✗ TensorFlow test failed: {str(e)}")
        return False


def test_kaggle_api():
    """Test Kaggle API configuration."""
    print("\nTesting Kaggle API...")
    
    kaggle_json = os.path.expanduser('~/.kaggle/kaggle.json')
    
    if os.path.exists(kaggle_json):
        print(f"  ✓ Kaggle credentials found at {kaggle_json}")
        
        # Check permissions (should be 600)
        import stat
        permissions = oct(os.stat(kaggle_json).st_mode)[-3:]
        if permissions == '600':
            print(f"  ✓ Permissions are correct (600)")
        else:
            print(f"  ⚠ Permissions are {permissions}, should be 600")
            print(f"    Run: chmod 600 {kaggle_json}")
        
        return True
    else:
        print(f"  ✗ Kaggle credentials not found at {kaggle_json}")
        print("  Please download kaggle.json from https://www.kaggle.com/settings")
        print(f"  and place it at {kaggle_json}")
        return False


def test_imports_from_src():
    """Test if project modules can be imported."""
    print("\nTesting project modules...")
    
    try:
        from src.data_loader import CROHME2019Dataset, InkMLParser
        print("  ✓ src.data_loader")
        
        from src.preprocessing import DataPreprocessor, DataGenerator
        print("  ✓ src.preprocessing")
        
        from src.model import create_model, HandwrittenEquationSolver
        print("  ✓ src.model")
        
        from src.utils import Vocabulary, normalize_strokes
        print("  ✓ src.utils")
        
        print("\n✅ All project modules imported successfully!")
        return True
    
    except Exception as e:
        print(f"\n❌ Failed to import project modules: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_vocabulary():
    """Test vocabulary functionality."""
    print("\nTesting vocabulary...")
    
    try:
        from src.utils import Vocabulary
        
        vocab = Vocabulary()
        
        # Test encoding
        test_string = "x + y = z"
        encoded = vocab.encode(test_string)
        decoded = vocab.decode(encoded)
        
        if decoded == test_string:
            print(f"  ✓ Vocabulary encode/decode works")
        else:
            print(f"  ✗ Vocabulary encode/decode failed")
            print(f"    Input:  '{test_string}'")
            print(f"    Output: '{decoded}'")
            return False
        
        print(f"  ✓ Vocabulary size: {len(vocab)}")
        return True
    
    except Exception as e:
        print(f"  ✗ Vocabulary test failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("Setup Test for Handwritten Equation Solver")
    print("="*80)
    
    tests = [
        ("Package Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("TensorFlow", test_tensorflow),
        ("Kaggle API", test_kaggle_api),
        ("Project Modules", test_imports_from_src),
        ("Vocabulary", test_vocabulary)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{test_name}' raised an exception: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("  1. Run: python example_usage.py")
        print("  2. Visualize: python visualize_data.py")
        print("  3. Train: python train.py --epochs 100 --batch_size 32")
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Set up Kaggle API: Place kaggle.json in ~/.kaggle/")
        print("  - Check project structure: Ensure all files are present")
    
    print("="*80)
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

