"""
Setup Verification Script
Checks that all dependencies and components are properly installed.
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.10+)")
        return False


def check_dependencies():
    """Check required dependencies."""
    print("\nChecking dependencies...")
    required = [
        'numpy',
        'pandas',
        'sklearn',
        'scipy',
        'matplotlib',
        'seaborn',
        'yaml',
        'pytest',
    ]
    
    optional = [
        'aeon',  # aeon-toolkit
        'sktime',
        'torch',
        'transformers',
        'peft',
        'bitsandbytes',
    ]
    
    all_good = True
    
    # Check required
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (required)")
            all_good = False
    
    # Check optional
    print("\nOptional dependencies (for full functionality):")
    for package in optional:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ⚠ {package} (optional, needed for training/LLM)")
    
    return all_good


def check_directory_structure():
    """Check that all required directories exist."""
    print("\nChecking directory structure...")
    required_dirs = [
        'data/loaders',
        'models/encoders',
        'training',
        'evaluation',
        'explainer',
        'utils',
        'experiments/configs',
        'tests',
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ (missing)")
            all_good = False
    
    return all_good


def check_key_files():
    """Check that key files exist."""
    print("\nChecking key files...")
    required_files = [
        'setup.py',
        'requirements.txt',
        'README.md',
        'data/loaders/slurry_loader.py',
        'models/rocket_heads.py',
        'models/fusion.py',
        'training/train_rocket.py',
        'explainer/llm_setup.py',
        'explainer/self_consistency.py',
        'evaluation/metrics.py',
        'tests/test_loaders.py',
        'experiments/configs/baseline.yaml',
    ]
    
    all_good = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
            all_good = False
    
    return all_good


def check_imports():
    """Check that local imports work."""
    print("\nChecking local imports...")
    imports_to_test = [
        ('data.loaders.slurry_loader', 'load_slurry_csv'),
        ('models.rocket_heads', 'MultiROCKETClassifier'),
        ('models.fusion', 'LateFusionClassifier'),
        ('utils.physical_checks', 'validate_window'),
        ('models.encoders.feature_extractor', 'extract_window_features'),
    ]
    
    all_good = True
    for module_name, obj_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[obj_name])
            obj = getattr(module, obj_name)
            print(f"  ✓ {module_name}.{obj_name}")
        except Exception as e:
            print(f"  ✗ {module_name}.{obj_name} ({str(e)[:50]})")
            all_good = False
    
    return all_good


def check_gpu_availability():
    """Check GPU availability for LLM."""
    print("\nChecking GPU availability (for LLM)...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ GPU available: {gpu_name}")
            print(f"    Memory: {gpu_memory:.1f} GB")
            if gpu_memory >= 8:
                print(f"    Sufficient for LLM (4-bit)")
            else:
                print(f"    ⚠ May be insufficient for LLM (recommend 8+ GB)")
        else:
            print(f"  ⚠ No GPU detected (LLM will not work)")
            print(f"    CPU-only mode available for classifier")
    except ImportError:
        print(f"  ⚠ PyTorch not installed (cannot check GPU)")


def main():
    """Run all verification checks."""
    print("="*60)
    print("FD-LLM SETUP VERIFICATION")
    print("="*60)
    
    results = []
    
    # Run checks
    results.append(("Python version", check_python_version()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Directory structure", check_directory_structure()))
    results.append(("Key files", check_key_files()))
    results.append(("Local imports", check_imports()))
    
    # GPU check (informational only)
    check_gpu_availability()
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for check_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {check_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    
    if all_passed:
        print("✓ All checks passed! System is ready.")
        print("\nNext steps:")
        print("  1. Place CSV data in data/raw/")
        print("  2. Run: python training/train_rocket.py --data data/raw/your_data.csv")
        print("  3. Run: python example_usage.py (for demonstration)")
        return 0
    else:
        print("✗ Some checks failed. Please resolve issues above.")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())

