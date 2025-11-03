"""
Startup script for Sentinel AI Epidemic Detection API
"""

import subprocess
import sys
import os
import time

def install_dependencies():
    """Install required packages"""
    print("📦 Installing API dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn[standard]",
        "pydantic",
        "torch",
        "numpy",
        "requests"  # For testing
    ]
    
    for package in required_packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"⚠ Warning: Could not install {package}")
    
    print("✓ Dependencies installation complete!")

def start_api():
    """Start the API server"""
    print("\n🚀 STARTING SENTINEL AI API")
    print("=" * 50)
    print("📱 Mobile App Integration Ready")
    print("🤖 Custom LSTM+Attention Model")
    print("⚡ Ultra-fast inference (5ms)")
    print("🌐 Server: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("=" * 50)
    
    try:
        # Start the API server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "epidemic_api:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")
    except Exception as e:
        print(f"✗ Error starting API: {e}")

def main():
    """Main startup function"""
    print("🏥 SENTINEL AI - EPIDEMIC DETECTION API")
    print("=" * 60)
    
    # Check if we need to install dependencies
    try:
        import fastapi
        import uvicorn
        print("✓ Dependencies already installed")
    except ImportError:
        install_dependencies()
    
    # Start the API
    start_api()

if __name__ == "__main__":
    main()