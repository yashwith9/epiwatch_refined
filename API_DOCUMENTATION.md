# 🏥 Sentinel AI - Epidemic Detection API

## Overview

This API provides real-time epidemic detection and monitoring capabilities using a Custom LSTM+Attention neural network model. It's designed to integrate seamlessly with the Sentinel AI mobile application.

## 🚀 Quick Start

### 1. Start the API Server
```bash
python start_api.py
```

### 2. Access the API
- **Base URL**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

### 3. Test the API
```bash
python test_api.py
```

## 📱 Mobile App Integration

The API provides all endpoints needed for your mobile app's 4 main tabs:

### 🔔 Alerts Tab
- `GET /alerts` - Get all alerts
- `GET /alerts/filter/{risk_level}` - Filter alerts by risk level
- `POST /alerts/investigate/{alert_id}` - Mark alert for investigation

### 🗺️ Map Tab
- `GET /map/data` - Get outbreak locations and risk zones
- Provides coordinates for map visualization
- Risk level indicators (Critical, Moderate, Low)

### 📈 Trends Tab
- `GET /trends` - Get epidemic trends and analytics
- Weekly disease progression data
- Disease breakdown with case counts
- Trend directions and percentage changes

### 👤 Profile Tab
- `GET /model/info` - Model performance statistics
- `GET /health` - System health status

## 🔍 Core Prediction Endpoints

### Single Text Prediction
```http
POST /predict
Content-Type: application/json

{
  "text": "Outbreak of dengue fever reported in Mumbai",
  "location": "Mumbai, India",
  "timestamp": "2025-01-01T10:00:00Z"
}
```

**Response:**
```json
{
  "prediction": 1,
  "confidence": 0.87,
  "risk_level": "high",
  "processing_time_ms": 5.2,
  "model_version": "Custom LSTM+Attention v1.0"
}
```

### Batch Prediction
```http
POST /predict/batch
Content-Type: application/json

{
  "texts": [
    "Dengue outbreak in Mumbai",
    "Weather is nice today",
    "COVID cases rising"
  ]
}
```

## 📊 Dashboard Data

### Dashboard Statistics
```http
GET /dashboard/stats
```

**Response:**
```json
{
  "total_cases": 8081,
  "countries_affected": 8,
  "critical_alerts": 2,
  "regions_monitored": 6,
  "network_status": "Global Network • 8 Outbreaks Tracked",
  "outbreaks_tracked": 8
}
```

## 📈 Trends and Analytics

### Epidemic Trends
```http
GET /trends
```

**Response includes:**
- Weekly progression data for 6 diseases
- Disease breakdown with case counts
- Trend directions (up/down/stable)
- Percentage changes

### Sample Response:
```json
{
  "weekly_data": [...],
  "disease_breakdown": [
    {
      "disease": "Dengue",
      "cases": 287,
      "trend_direction": "up",
      "percentage_change": 15.2
    }
  ],
  "total_cases_trend": [150, 180, 210, 240, 280, 320]
}
```

## 🗺️ Map Data

### Outbreak Locations
```http
GET /map/data
```

**Response includes:**
- Active outbreak locations with coordinates
- Risk zones by region
- Global status summary

### Sample Response:
```json
{
  "active_outbreaks": [
    {
      "disease": "Dengue Fever",
      "location": "Mumbai",
      "country": "India",
      "region": "Asia",
      "cases": 2847,
      "risk_level": "critical",
      "coordinates": {"lat": 19.0760, "lng": 72.8777}
    }
  ],
  "risk_zones": [...],
  "global_status": "8 active • 8 countries"
}
```

## 🔔 Alerts System

### Get All Alerts
```http
GET /alerts
```

### Filter Alerts by Risk Level
```http
GET /alerts/filter/high
GET /alerts/filter/moderate
GET /alerts/filter/low
GET /alerts/filter/all
```

### Sample Alert Response:
```json
{
  "alerts": [
    {
      "id": "ALERT_001",
      "title": "Dengue Fever Surge Detected",
      "description": "Unusual spike in dengue cases...",
      "disease": "Dengue Fever",
      "location": "Mumbai",
      "country": "India",
      "region": "Asia",
      "risk_level": "high",
      "timestamp": "47 minutes ago",
      "cases": 2847,
      "action_required": true
    }
  ],
  "total_alerts": 3,
  "critical_count": 2,
  "moderate_count": 1,
  "low_count": 0
}
```

## 🤖 Model Information

### Model Performance
```http
GET /model/info
```

**Response:**
```json
{
  "model_name": "Custom LSTM+Attention",
  "version": "1.0.0",
  "architecture": "Embedding -> Bi-LSTM -> Attention -> Dense -> Sigmoid",
  "performance": {
    "accuracy": 0.82,
    "f1_score": 0.81,
    "inference_speed_ms": 5.0,
    "training_time_s": 1.0
  }
}
```

## 🏥 Health Monitoring

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T10:00:00Z",
  "model_loaded": true,
  "vocab_loaded": true,
  "device": "cpu",
  "memory_usage": "Normal",
  "api_version": "1.0.0"
}
```

## 🔧 Technical Details

### Model Architecture
- **Type**: Custom LSTM+Attention Neural Network
- **Architecture**: Embedding → Bi-LSTM → Attention → Dense → Sigmoid
- **Performance**: 82% accuracy, 0.81 F1-score
- **Speed**: 5ms inference time per sample
- **Training**: 1 second (345x faster than transformers)

### API Features
- **FastAPI** framework for high performance
- **Pydantic** models for data validation
- **CORS** enabled for mobile app integration
- **Automatic documentation** at `/docs`
- **Health monitoring** and error handling
- **Batch processing** support

### Response Times
- Single prediction: ~5ms
- Batch prediction: ~5ms per text
- Dashboard stats: <1ms
- Trends data: <1ms
- Map data: <1ms
- Alerts: <1ms

## 📱 Mobile App Integration Guide

### 1. Dashboard Tab Integration
```javascript
// Fetch dashboard stats
const response = await fetch('http://localhost:8000/dashboard/stats');
const stats = await response.json();

// Update UI
document.getElementById('total-cases').textContent = stats.total_cases;
document.getElementById('countries').textContent = stats.countries_affected;
document.getElementById('critical-alerts').textContent = stats.critical_alerts;
document.getElementById('regions').textContent = stats.regions_monitored;
```

### 2. Trends Tab Integration
```javascript
// Fetch trends data
const response = await fetch('http://localhost:8000/trends');
const trends = await response.json();

// Update charts
updateLineChart(trends.total_cases_trend);
updateDiseaseBreakdown(trends.disease_breakdown);
```

### 3. Map Tab Integration
```javascript
// Fetch map data
const response = await fetch('http://localhost:8000/map/data');
const mapData = await response.json();

// Add markers to map
mapData.active_outbreaks.forEach(outbreak => {
  addMarker(outbreak.coordinates, outbreak.risk_level, outbreak.disease);
});
```

### 4. Alerts Tab Integration
```javascript
// Fetch alerts
const response = await fetch('http://localhost:8000/alerts');
const alerts = await response.json();

// Filter by risk level
const criticalAlerts = await fetch('http://localhost:8000/alerts/filter/high');
```

## 🚀 Deployment

### Local Development
```bash
python start_api.py
```

### Production Deployment
```bash
uvicorn epidemic_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "epidemic_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🔒 Security Considerations

- Enable HTTPS in production
- Implement API key authentication
- Add rate limiting
- Validate all inputs
- Monitor for abuse

## 📞 Support

For technical support or questions about the API:
- Check the interactive documentation at `/docs`
- Run the test suite with `python test_api.py`
- Review the model performance metrics at `/model/info`

## 🎯 Performance Benchmarks

- **Inference Speed**: 5ms per prediction (200 predictions/second)
- **Batch Processing**: 5ms per text in batch
- **Memory Usage**: ~100MB for model
- **CPU Usage**: <10% on modern hardware
- **Accuracy**: 82% on test data
- **F1-Score**: 0.81

---

**🏆 Winner Model**: Custom LSTM+Attention chosen for its perfect balance of speed and accuracy!