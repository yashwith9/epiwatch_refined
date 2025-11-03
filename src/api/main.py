"""
FastAPI Backend for EpiWatch Mobile Application
Serves model predictions, alerts, and visualizations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import os

app = FastAPI(
    title="EpiWatch API",
    description="AI-powered epidemic detection system for low-resource regions",
    version="1.0.0"
)

# CORS middleware for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== DATA MODELS =====

class TextInput(BaseModel):
    """Input model for text classification"""
    text: str
    region: Optional[str] = None
    disease: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    text: str
    prediction: int
    probability: float
    is_outbreak: bool
    confidence: str


class Alert(BaseModel):
    """Alert model"""
    id: int
    title: str
    location: str
    risk_level: str
    case_count: int
    date: str
    summary: str
    color: str


class TrendData(BaseModel):
    """Trend data model"""
    disease: str
    data: List[dict]


class MapRegion(BaseModel):
    """Map region data"""
    region: str
    risk_level: str
    alert_count: int
    color: str


# ===== GLOBAL STATE =====

# This would typically be loaded from database or file system
CURRENT_ALERTS = []
TREND_DATA = {}
MAP_DATA = []


# ===== UTILITY FUNCTIONS =====

def load_model():
    """Load the best trained model"""
    # TODO: Implement actual model loading
    # For now, return None (will be implemented after training)
    return None


def load_alerts():
    """Load current alerts from file"""
    alerts_file = "outputs/alerts/current_alerts.json"
    if os.path.exists(alerts_file):
        with open(alerts_file, 'r') as f:
            return json.load(f)
    return []


def generate_sample_alerts():
    """Generate sample alerts for demonstration"""
    return [
        {
            "id": 1,
            "title": "Dengue Fever Alert",
            "location": "Mumbai, India",
            "risk_level": "high",
            "case_count": 287,
            "date": datetime.now().isoformat(),
            "summary": "Unusual spike in dengue cases reported across Mumbai region. Health authorities investigating. Immediate action recommended.",
            "color": "#FF4444"
        },
        {
            "id": 2,
            "title": "Malaria Cases Increasing",
            "location": "Nairobi, Kenya",
            "risk_level": "moderate",
            "case_count": 134,
            "date": (datetime.now() - timedelta(hours=12)).isoformat(),
            "summary": "Rising malaria cases detected in Nairobi area with vector activity above normal levels.",
            "color": "#FFA500"
        },
        {
            "id": 3,
            "title": "Cholera Warning Signal",
            "location": "Dhaka, Bangladesh",
            "risk_level": "moderate",
            "case_count": 67,
            "date": (datetime.now() - timedelta(hours=3)).isoformat(),
            "summary": "Early indication suggest potential cholera outbreak risk in Dhaka region.",
            "color": "#FFA500"
        }
    ]


def generate_sample_trends():
    """Generate sample 7-day trend data"""
    diseases = ["Dengue", "Malaria", "Cholera", "Yellow Fever", "Measles", "Influenza"]
    trends = {}
    
    for disease in diseases:
        data = []
        base = np.random.randint(20, 100)
        
        for i in range(7):
            date = datetime.now() - timedelta(days=6-i)
            count = base + np.random.randint(-20, 40)
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "count": max(0, count)
            })
        
        trends[disease] = {
            "disease": disease,
            "data": data
        }
    
    return trends


def generate_sample_map_data():
    """Generate sample map data"""
    return [
        {"region": "Mumbai", "risk_level": "high", "alert_count": 3, "color": "#FF4444"},
        {"region": "Nairobi", "risk_level": "moderate", "alert_count": 2, "color": "#FFA500"},
        {"region": "Dhaka", "risk_level": "moderate", "alert_count": 2, "color": "#FFA500"},
        {"region": "Delhi", "risk_level": "low", "alert_count": 1, "color": "#4CAF50"},
        {"region": "Lagos", "risk_level": "low", "alert_count": 1, "color": "#4CAF50"},
    ]


# ===== API ENDPOINTS =====

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to EpiWatch API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "alerts": "/alerts",
            "trends": "/trends",
            "map": "/map",
            "detect": "/detect",
            "health": "/health",
            "stats": "/stats"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected",  # TODO: Check actual DB connection
        "model": "loaded"  # TODO: Check if model is loaded
    }


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    alerts = generate_sample_alerts()
    
    return {
        "total_cases": 8081,
        "countries": 8,
        "critical_alerts": 2,
        "regions_monitored": 6,
        "active_alerts": len(alerts),
        "last_update": datetime.now().isoformat()
    }


@app.get("/alerts", response_model=List[Alert])
async def get_alerts(
    risk_level: Optional[str] = None,
    limit: Optional[int] = None
):
    """
    Get current outbreak alerts
    
    Args:
        risk_level: Filter by risk level (high, moderate, low)
        limit: Maximum number of alerts to return
    
    Returns:
        List of alerts
    """
    # Load or generate alerts
    alerts = generate_sample_alerts()
    
    # Filter by risk level
    if risk_level:
        alerts = [a for a in alerts if a['risk_level'] == risk_level.lower()]
    
    # Limit results
    if limit:
        alerts = alerts[:limit]
    
    return alerts


@app.get("/trends", response_model=dict)
async def get_trends(days: int = 7):
    """
    Get disease trend data
    
    Args:
        days: Number of days of trend data (default: 7)
    
    Returns:
        Dictionary with trend data for each disease
    """
    trends = generate_sample_trends()
    return trends


@app.get("/map", response_model=List[MapRegion])
async def get_map_data():
    """
    Get geospatial outbreak data for map visualization
    
    Returns:
        List of regions with risk levels
    """
    map_data = generate_sample_map_data()
    return map_data


@app.post("/detect", response_model=PredictionResponse)
async def detect_outbreak(input_data: TextInput):
    """
    Classify text for outbreak signals
    
    Args:
        input_data: Text input with optional metadata
    
    Returns:
        Prediction with probability and confidence
    """
    # TODO: Load actual model and make prediction
    # For now, return dummy prediction based on keywords
    
    text = input_data.text.lower()
    outbreak_keywords = ['outbreak', 'epidemic', 'fever', 'cases', 'death', 
                        'infection', 'disease', 'sick', 'hospital']
    
    # Simple keyword-based prediction (replace with actual model)
    keyword_count = sum(1 for word in outbreak_keywords if word in text)
    probability = min(keyword_count * 0.15, 0.95)
    prediction = 1 if probability > 0.5 else 0
    
    # Determine confidence level
    if probability > 0.8:
        confidence = "high"
    elif probability > 0.5:
        confidence = "medium"
    else:
        confidence = "low"
    
    return {
        "text": input_data.text,
        "prediction": prediction,
        "probability": float(probability),
        "is_outbreak": bool(prediction),
        "confidence": confidence
    }


@app.get("/diseases")
async def get_diseases():
    """
    Get list of tracked diseases
    
    Returns:
        List of disease names with statistics
    """
    return {
        "diseases": [
            {"name": "Dengue", "cases": 287, "trend": "up"},
            {"name": "Malaria", "cases": 134, "trend": "stable"},
            {"name": "Cholera", "cases": 67, "trend": "up"},
            {"name": "Yellow Fever", "cases": 73, "trend": "down"},
            {"name": "Measles", "cases": 68, "trend": "stable"},
            {"name": "Influenza", "cases": 103, "trend": "up"}
        ]
    }


@app.get("/regions")
async def get_regions():
    """
    Get list of monitored regions
    
    Returns:
        List of regions with statistics
    """
    return {
        "regions": [
            {"name": "Mumbai, India", "alerts": 3, "risk": "high"},
            {"name": "Nairobi, Kenya", "alerts": 2, "risk": "moderate"},
            {"name": "Dhaka, Bangladesh", "alerts": 2, "risk": "moderate"},
            {"name": "Delhi, India", "alerts": 1, "risk": "low"},
            {"name": "Lagos, Nigeria", "alerts": 1, "risk": "low"},
            {"name": "Manila, Philippines", "alerts": 1, "risk": "low"}
        ]
    }


@app.get("/alert/{alert_id}")
async def get_alert_detail(alert_id: int):
    """
    Get detailed information about a specific alert
    
    Args:
        alert_id: Alert ID
    
    Returns:
        Detailed alert information
    """
    alerts = generate_sample_alerts()
    
    alert = next((a for a in alerts if a['id'] == alert_id), None)
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    # Add additional details
    alert['details'] = {
        "affected_areas": ["Downtown", "Suburbs", "Rural districts"],
        "symptoms": ["Fever", "Headache", "Joint pain"],
        "recommended_actions": [
            "Avoid stagnant water",
            "Use mosquito repellent",
            "Seek medical attention if symptomatic"
        ],
        "last_updated": datetime.now().isoformat()
    }
    
    return alert


# ===== STARTUP EVENT =====

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    print("="*60)
    print("🚀 EpiWatch API Starting...")
    print("="*60)
    print("✓ Loading models...")
    # TODO: Load trained model
    print("✓ Connecting to database...")
    # TODO: Connect to MongoDB
    print("✓ Loading alerts...")
    # TODO: Load current alerts
    print("="*60)
    print("✅ EpiWatch API is ready!")
    print("📍 API Documentation: http://localhost:8000/docs")
    print("="*60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
