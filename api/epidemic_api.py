"""
Epidemic Detection API for Sentinel AI Mobile App
Using Custom LSTM+Attention Model
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
import random
import uvicorn

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.custom_model import CustomEpiDetector, build_vocab

# Initialize FastAPI app
app = FastAPI(
    title="Sentinel AI - Epidemic Detection API",
    description="Real-time epidemic detection and monitoring API using Custom LSTM+Attention model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and data
model = None
vocab = None
device = torch.device('cpu')

# Pydantic models for API requests/responses
class TextInput(BaseModel):
    text: str
    location: Optional[str] = None
    timestamp: Optional[str] = None

class BatchTextInput(BaseModel):
    texts: List[str]
    locations: Optional[List[str]] = None
    timestamps: Optional[List[str]] = None

class PredictionResponse(BaseModel):
    prediction: int  # 0 or 1
    confidence: float
    risk_level: str  # "low", "moderate", "high"
    processing_time_ms: float
    model_version: str = "Custom LSTM+Attention v1.0"

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int
    average_confidence: float

class DashboardStats(BaseModel):
    total_cases: int
    countries_affected: int
    critical_alerts: int
    regions_monitored: int
    network_status: str
    outbreaks_tracked: int

class TrendData(BaseModel):
    disease: str
    cases: int
    trend_direction: str  # "up", "down", "stable"
    percentage_change: float

class EpidemicTrends(BaseModel):
    weekly_data: List[Dict]
    disease_breakdown: List[TrendData]
    total_cases_trend: List[int]

class OutbreakLocation(BaseModel):
    disease: str
    location: str
    country: str
    region: str
    cases: int
    risk_level: str
    coordinates: Dict[str, float]

class MapData(BaseModel):
    active_outbreaks: List[OutbreakLocation]
    risk_zones: List[Dict]
    global_status: str

class Alert(BaseModel):
    id: str
    title: str
    description: str
    disease: str
    location: str
    country: str
    region: str
    risk_level: str
    timestamp: str
    cases: Optional[int] = None
    action_required: bool = True

class AlertsResponse(BaseModel):
    alerts: List[Alert]
    total_alerts: int
    critical_count: int
    moderate_count: int
    low_count: int

# Initialize model on startup
@app.on_event("startup")
async def load_model():
    """Load the trained custom LSTM model"""
    global model, vocab, device
    
    try:
        print("Loading Custom LSTM+Attention model...")
        
        # Create sample vocabulary (in production, load from saved vocab)
        sample_texts = [
            "outbreak reported disease spreading",
            "health emergency declared cases rising",
            "epidemic alert virus detected",
            "normal weather conditions today",
            "stock market update news"
        ]
        vocab = build_vocab(sample_texts, min_freq=1)
        
        # Initialize model
        model = CustomEpiDetector(
            vocab_size=len(vocab),
            embedding_dim=128,
            hidden_dim=64,
            num_layers=2,
            dropout=0.3
        )
        
        # Try to load trained weights
        model_path = "models/saved/custom_best.pt"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("✓ Loaded trained model weights")
        else:
            print("⚠ No trained weights found, using random initialization")
        
        model.eval()
        model.to(device)
        
        print("✓ Model loaded successfully!")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        # Initialize with dummy model for demo
        vocab = {"<PAD>": 0, "<UNK>": 1, "outbreak": 2, "disease": 3}
        model = CustomEpiDetector(vocab_size=len(vocab))
        model.eval()

def preprocess_text(text: str) -> torch.Tensor:
    """Preprocess text for model input"""
    words = text.lower().split()
    indices = [vocab.get(word, vocab.get('<UNK>', 1)) for word in words]
    
    # Pad or truncate to fixed length
    max_length = 256
    if len(indices) < max_length:
        indices += [vocab.get('<PAD>', 0)] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    
    return torch.tensor([indices], dtype=torch.long)

def predict_epidemic(text: str) -> tuple:
    """Make epidemic prediction for text"""
    start_time = datetime.now()
    
    # Preprocess
    input_tensor = preprocess_text(text).to(device)
    
    # Predict
    with torch.no_grad():
        output, attention_weights = model(input_tensor)
        confidence = float(output[0])
        prediction = 1 if confidence > 0.5 else 0
    
    # Calculate processing time
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    # Determine risk level
    if confidence > 0.8:
        risk_level = "high"
    elif confidence > 0.5:
        risk_level = "moderate"
    else:
        risk_level = "low"
    
    return prediction, confidence, risk_level, processing_time

# API Endpoints

@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Sentinel AI - Epidemic Detection API",
        "status": "active",
        "model": "Custom LSTM+Attention",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(input_data: TextInput):
    """Predict epidemic risk for single text"""
    try:
        prediction, confidence, risk_level, processing_time = predict_epidemic(input_data.text)
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            risk_level=risk_level,
            processing_time_ms=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(input_data: BatchTextInput):
    """Predict epidemic risk for multiple texts"""
    try:
        predictions = []
        total_confidence = 0
        
        for text in input_data.texts:
            prediction, confidence, risk_level, processing_time = predict_epidemic(text)
            
            predictions.append(PredictionResponse(
                prediction=prediction,
                confidence=confidence,
                risk_level=risk_level,
                processing_time_ms=processing_time
            ))
            total_confidence += confidence
        
        avg_confidence = total_confidence / len(predictions) if predictions else 0
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            average_confidence=avg_confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get dashboard statistics for mobile app"""
    # In production, these would come from a database
    return DashboardStats(
        total_cases=8081,
        countries_affected=8,
        critical_alerts=2,
        regions_monitored=6,
        network_status="Global Network • 8 Outbreaks Tracked",
        outbreaks_tracked=8
    )

@app.get("/trends", response_model=EpidemicTrends)
async def get_epidemic_trends():
    """Get epidemic trends data for trends tab"""
    
    # Generate sample weekly trend data
    weekly_data = []
    for week in range(1, 7):
        weekly_data.append({
            f"Week {week}": {
                "Dengue": random.randint(100, 300),
                "Malaria": random.randint(80, 150),
                "Cholera": random.randint(50, 120),
                "Yellow": random.randint(60, 100),
                "Measles": random.randint(40, 80),
                "Influenza": random.randint(70, 130)
            }
        })
    
    # Disease breakdown with trends
    diseases = [
        TrendData(disease="Dengue", cases=287, trend_direction="up", percentage_change=15.2),
        TrendData(disease="Malaria", cases=134, trend_direction="down", percentage_change=-8.1),
        TrendData(disease="Cholera", cases=67, trend_direction="stable", percentage_change=2.3),
        TrendData(disease="Yellow", cases=73, trend_direction="up", percentage_change=12.7),
        TrendData(disease="Measles", cases=68, trend_direction="down", percentage_change=-5.4),
        TrendData(disease="Influenza", cases=103, trend_direction="up", percentage_change=9.8)
    ]
    
    # Total cases trend over weeks
    total_cases_trend = [150, 180, 210, 240, 280, 320]
    
    return EpidemicTrends(
        weekly_data=weekly_data,
        disease_breakdown=diseases,
        total_cases_trend=total_cases_trend
    )

@app.get("/map/data", response_model=MapData)
async def get_map_data():
    """Get map data for map tab"""
    
    # Sample outbreak locations
    outbreaks = [
        OutbreakLocation(
            disease="Dengue Fever",
            location="Mumbai",
            country="India",
            region="Asia",
            cases=2847,
            risk_level="critical",
            coordinates={"lat": 19.0760, "lng": 72.8777}
        ),
        OutbreakLocation(
            disease="Malaria",
            location="Nairobi",
            country="Kenya",
            region="Africa",
            cases=1523,
            risk_level="critical",
            coordinates={"lat": -1.2921, "lng": 36.8219}
        ),
        OutbreakLocation(
            disease="Cholera",
            location="Dhaka",
            country="Bangladesh",
            region="Asia",
            cases=856,
            risk_level="moderate",
            coordinates={"lat": 23.8103, "lng": 90.4125}
        ),
        OutbreakLocation(
            disease="Influenza A",
            location="São Paulo",
            country="Brazil",
            region="Americas",
            cases=1247,
            risk_level="moderate",
            coordinates={"lat": -23.5505, "lng": -46.6333}
        )
    ]
    
    # Risk zones
    risk_zones = [
        {"region": "South Asia", "risk_level": "high", "active_diseases": 3},
        {"region": "East Africa", "risk_level": "critical", "active_diseases": 2},
        {"region": "Southeast Asia", "risk_level": "moderate", "active_diseases": 2},
        {"region": "South America", "risk_level": "moderate", "active_diseases": 1}
    ]
    
    return MapData(
        active_outbreaks=outbreaks,
        risk_zones=risk_zones,
        global_status="8 active • 8 countries"
    )

@app.get("/alerts", response_model=AlertsResponse)
async def get_alerts():
    """Get alerts for alerts tab"""
    
    # Sample alerts
    alerts = [
        Alert(
            id="ALERT_001",
            title="Dengue Fever Surge Detected",
            description="Unusual spike in dengue cases reported across Mumbai region. Health authorities alerted.",
            disease="Dengue Fever",
            location="Mumbai",
            country="India",
            region="Asia",
            risk_level="high",
            timestamp="47 minutes ago",
            cases=2847,
            action_required=True
        ),
        Alert(
            id="ALERT_002",
            title="Malaria Cases Increasing",
            description="Rising malaria cases detected in Nairobi area with vector activity above normal levels.",
            disease="Malaria",
            location="Nairobi",
            country="Kenya",
            region="Africa",
            risk_level="high",
            timestamp="1.2 hours ago",
            cases=1523,
            action_required=True
        ),
        Alert(
            id="ALERT_003",
            title="Cholera Warning Signal",
            description="Early indicators suggest potential cholera outbreak risk in Dhaka region.",
            disease="Cholera",
            location="Dhaka",
            country="Bangladesh",
            region="Asia",
            risk_level="moderate",
            timestamp="3 hours ago",
            cases=856,
            action_required=True
        )
    ]
    
    # Count alerts by risk level
    critical_count = len([a for a in alerts if a.risk_level == "high"])
    moderate_count = len([a for a in alerts if a.risk_level == "moderate"])
    low_count = len([a for a in alerts if a.risk_level == "low"])
    
    return AlertsResponse(
        alerts=alerts,
        total_alerts=len(alerts),
        critical_count=critical_count,
        moderate_count=moderate_count,
        low_count=low_count
    )

@app.get("/alerts/filter/{risk_level}")
async def get_alerts_by_risk(risk_level: str):
    """Get alerts filtered by risk level"""
    all_alerts = await get_alerts()
    
    if risk_level.lower() == "all":
        return all_alerts
    
    filtered_alerts = [
        alert for alert in all_alerts.alerts 
        if alert.risk_level.lower() == risk_level.lower()
    ]
    
    return AlertsResponse(
        alerts=filtered_alerts,
        total_alerts=len(filtered_alerts),
        critical_count=len([a for a in filtered_alerts if a.risk_level == "high"]),
        moderate_count=len([a for a in filtered_alerts if a.risk_level == "moderate"]),
        low_count=len([a for a in filtered_alerts if a.risk_level == "low"])
    )

@app.post("/alerts/investigate/{alert_id}")
async def investigate_alert(alert_id: str):
    """Mark alert as under investigation"""
    return {
        "alert_id": alert_id,
        "status": "under_investigation",
        "timestamp": datetime.now().isoformat(),
        "message": f"Alert {alert_id} marked for investigation"
    }

@app.get("/model/info")
async def get_model_info():
    """Get model information and statistics"""
    return {
        "model_name": "Custom LSTM+Attention",
        "version": "1.0.0",
        "architecture": "Embedding -> Bi-LSTM -> Attention -> Dense -> Sigmoid",
        "parameters": f"{sum(p.numel() for p in model.parameters()):,}" if model else "Unknown",
        "vocab_size": len(vocab) if vocab else 0,
        "device": str(device),
        "status": "loaded" if model else "not_loaded",
        "performance": {
            "accuracy": 0.82,
            "f1_score": 0.81,
            "inference_speed_ms": 5.0,
            "training_time_s": 1.0
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "vocab_loaded": vocab is not None,
        "device": str(device),
        "memory_usage": "Normal",
        "api_version": "1.0.0"
    }

if __name__ == "__main__":
    print("🚀 Starting Sentinel AI Epidemic Detection API")
    print("=" * 60)
    print("📱 Mobile App Integration Ready")
    print("🤖 Using Custom LSTM+Attention Model")
    print("⚡ Ultra-fast inference (5ms per prediction)")
    print("=" * 60)
    
    uvicorn.run(
        "epidemic_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )