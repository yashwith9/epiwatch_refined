"""
Setup script to copy dataset and run training
"""

import os
import shutil
import subprocess
import sys

def setup_dataset():
    """Copy dataset to workspace"""
    source_path = r"C:\Users\Bruger\Downloads\disease_outbreaks_minimal.csv"
    target_path = "data/disease_outbreaks_minimal.csv"
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    try:
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            print(f"✓ Dataset copied to {target_path}")
            return target_path
        else:
            print(f"✗ Dataset not found at {source_path}")
            return None
    except Exception as e:
        print(f"✗ Error copying dataset: {e}")
        return None

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed")
    except Exception as e:
        print(f"✗ Error installing requirements: {e}")

def run_training():
    """Run the training script"""
    print("Starting model training...")
    try:
        subprocess.run([sys.executable, "train_five_models_fast.py"])
    except Exception as e:
        print(f"✗ Error running training: {e}")

if __name__ == "__main__":
    print("="*60)
    print("EPIWATCH MODEL TRAINING SETUP")
    print("="*60)
    
    # Setup dataset
    dataset_path = setup_dataset()
    
    if dataset_path:
        # Install requirements
        install_requirements()
        
        # Run training
        run_training()
    else:
        print("Please ensure your dataset is available and try again.")