#!/usr/bin/env python3
"""
Quick Setup and Test Script
Run this first to verify your environment is ready
"""

import sys
import subprocess


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print("✅ Python version OK")
    return True


def install_requirements():
    """Install required packages"""
    print("\n" + "=" * 60)
    print("Installing required packages...")
    print("=" * 60)
    
    packages = [
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'scipy'
    ]
    
    try:
        for package in packages:
            print(f"\nInstalling {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        
        print("\n✅ All packages installed successfully!")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error installing packages: {e}")
        return False


def test_imports():
    """Test if all imports work"""
    print("\n" + "=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    imports = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
    ]
    
    for module, alias in imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            return False
    
    return True


def verify_datasets():
    """Check if datasets exist"""
    print("\n" + "=" * 60)
    print("Checking datasets...")
    print("=" * 60)
    
    import os
    
    datasets = [
        'ev_charging_dataset.csv',
        'location_dataset.csv'
    ]
    
    all_exist = True
    for dataset in datasets:
        if os.path.exists(dataset):
            size = os.path.getsize(dataset) / (1024 * 1024)  # MB
            print(f"✅ {dataset} ({size:.2f} MB)")
        else:
            print(f"❌ {dataset} not found")
            all_exist = False
    
    return all_exist


def run_quick_test():
    """Run a quick test of core functionality"""
    print("\n" + "=" * 60)
    print("Running quick functionality test...")
    print("=" * 60)
    
    try:
        import numpy as np
        import pandas as pd
        
        # Test data loading
        print("\nTesting data loading...")
        df = pd.read_csv('ev_charging_dataset.csv', nrows=100)
        print(f"✅ Loaded {len(df)} sample records")
        
        # Test basic computation
        print("\nTesting numpy operations...")
        arr = np.random.rand(1000, 1000)
        result = np.dot(arr, arr.T)
        print(f"✅ Numpy computation successful")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


def main():
    """Main setup function"""
    print("=" * 60)
    print("EV CHARGING OPTIMIZATION - SETUP SCRIPT")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    # Verify datasets
    if not verify_datasets():
        print("\n⚠️  Warning: Some datasets are missing")
        print("Make sure ev_charging_dataset.csv is in the current directory")
    
    # Run quick test
    if not run_quick_test():
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print("\nYou're ready to run the project:")
    print("\n  python main.py")
    print("\nThis will execute the entire pipeline (15-30 minutes).")
    print("\nResults will be saved to the 'results/' directory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
