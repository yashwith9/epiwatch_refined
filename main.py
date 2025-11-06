"""
EpiWatch API - Main entry point for Render deployment
Updated: 2025-11-06 - Fixed trends endpoint with random module import
"""
import sys
import os

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the FastAPI app
from api.main import app

# This is the app that Render will run with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
