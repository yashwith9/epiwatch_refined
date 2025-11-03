# EpiWatch - Quick Start Guide

## 🚀 Getting Started

### 1. Environment Setup

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 2. Project Structure

```
NLP/
├── data/
│   ├── raw/              # Raw scraped data
│   └── processed/        # Preprocessed training data
├── models/
│   ├── scratch/          # Custom neural network
│   ├── pretrained/       # Fine-tuned transformer models
│   └── saved/            # Trained model checkpoints
├── src/
│   ├── data_collection/  # Data scraping (GDELT, NewsAPI)
│   ├── preprocessing/    # Text preprocessing pipeline
│   ├── models/           # Model implementations
│   │   ├── custom_model.py         # Custom LSTM + Attention
│   │   ├── pretrained_models.py    # Transformer models
│   │   └── train_all.py            # Complete training pipeline
│   ├── evaluation/       # Model comparison & metrics
│   │   ├── model_evaluator.py      # Comprehensive evaluation
│   │   └── anomaly_detection.py    # Outbreak alert system
│   └── api/              # FastAPI backend
│       └── main.py                 # REST API endpoints
├── notebooks/            # Jupyter notebooks
│   └── epiwatch_training.ipynb    # Interactive training notebook
├── outputs/              # Generated results
│   ├── alerts/          # Alert JSON files
│   └── visualizations/  # Performance plots & charts
└── config/
    └── config.yaml      # Configuration settings
```

## 📊 Training Models

### Option 1: Use Jupyter Notebook (Recommended for Learning)

```powershell
# Start Jupyter
jupyter notebook

# Open: notebooks/epiwatch_training.ipynb
# Follow step-by-step instructions
```

### Option 2: Run Complete Training Pipeline

```powershell
# Train all 5 models at once
python src/models/train_all.py
```

This will:
- ✅ Load and preprocess data
- ✅ Train custom neural network from scratch
- ✅ Fine-tune 4 pre-trained transformer models:
  - XLM-RoBERTa
  - mBERT (Multilingual BERT)
  - DistilBERT
  - MuRIL (for Indian languages)
- ✅ Compare all models
- ✅ Generate visualizations
- ✅ Recommend best model

### Expected Output

```
outputs/
├── model_comparison_results.json    # Detailed metrics
├── model_comparison_table.csv       # Comparison table
├── recommendation.json              # Best model recommendation
├── visualizations/
│   ├── model_comparison.png        # Performance comparison
│   └── confusion_matrices.png      # All confusion matrices
└── alerts/
    └── current_alerts.json         # Sample alerts
```

## 🌐 Starting the API

```powershell
# Start FastAPI server
uvicorn src.api.main:app --reload

# API will be available at: http://localhost:8000
# Documentation: http://localhost:8000/docs
```

### API Endpoints for Mobile App

1. **GET /alerts** - Get current outbreak alerts
   - Response: List of alerts with risk levels

2. **GET /trends?days=7** - Get 7-day disease trends
   - Response: Time-series data for each disease

3. **GET /map** - Get geospatial outbreak data
   - Response: Region data with risk levels for map

4. **POST /detect** - Classify new text for outbreak signals
   - Request: `{"text": "Multiple dengue cases reported..."}`
   - Response: Prediction with probability

5. **GET /stats** - Get system statistics
   - Response: Total cases, countries, alerts, etc.

## 📱 Mobile App Integration

### 3 Main Screens Data

#### 1. Main Dashboard (Map + Summary)
```javascript
// Fetch map data
fetch('http://localhost:8000/map')
  .then(res => res.json())
  .then(data => renderMap(data));

// Fetch stats
fetch('http://localhost:8000/stats')
  .then(res => res.json())
  .then(data => updateSummaryCards(data));
```

#### 2. Recent Alerts Feed
```javascript
// Fetch alerts (high priority first)
fetch('http://localhost:8000/alerts?limit=20')
  .then(res => res.json())
  .then(alerts => renderAlertFeed(alerts));
```

#### 3. 7-Day Trends
```javascript
// Fetch trend data
fetch('http://localhost:8000/trends?days=7')
  .then(res => res.json())
  .then(trends => renderTrendCharts(trends));
```

## 📈 Model Evaluation Metrics

The system evaluates models on:

1. **Accuracy** - Overall correctness
2. **Precision** - Avoiding false alarms
3. **Recall** - Detecting actual outbreaks
4. **F1-Score** - Balance between precision & recall
5. **AUC-ROC** - Classification performance
6. **Inference Time** - Speed (important for mobile)
7. **Model Size** - Memory footprint

## 🎯 Expected Model Performance

Based on similar epidemic detection tasks:

| Model | Expected F1 | Speed | Size |
|-------|------------|-------|------|
| Custom LSTM | 0.75-0.82 | Fast | Small |
| XLM-RoBERTa | 0.85-0.92 | Medium | Large |
| mBERT | 0.83-0.90 | Medium | Large |
| DistilBERT | 0.80-0.87 | **Fast** | Medium |
| MuRIL | 0.84-0.91 | Medium | Large |

**Recommended:** XLM-RoBERTa or mBERT for best accuracy, DistilBERT for production (faster inference).

## 🔍 Testing the System

### Test 1: Text Classification
```powershell
# Test the API
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: application/json" \
  -d '{"text": "Multiple cases of dengue fever reported in Mumbai region", "region": "Mumbai", "disease": "Dengue"}'
```

Expected Response:
```json
{
  "text": "Multiple cases of dengue fever reported in Mumbai region",
  "prediction": 1,
  "probability": 0.89,
  "is_outbreak": true,
  "confidence": "high"
}
```

### Test 2: Get Alerts
```powershell
curl "http://localhost:8000/alerts?risk_level=high"
```

## 🐛 Troubleshooting

### CUDA Out of Memory
```powershell
# Reduce batch size in train_all.py or notebook
# Change: batch_size=16 → batch_size=8
```

### NLTK Data Missing
```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Transformers Model Download Issues
```powershell
# Set cache directory
$env:TRANSFORMERS_CACHE="C:\Users\Bruger\.cache\huggingface"
```

## 📚 Additional Resources

- **Hugging Face Transformers**: https://huggingface.co/docs/transformers
- **FastAPI Documentation**: https://fastapi.tiangolo.com
- **PyTorch Tutorials**: https://pytorch.org/tutorials
- **GDELT Project**: https://www.gdeltproject.org

## 🌟 Next Steps After Training

1. ✅ Train all 5 models
2. ✅ Select best model based on metrics
3. 🔄 Deploy API to cloud (AWS/Azure/GCP)
4. 🔄 Connect Flutter/React Native mobile app
5. 🔄 Set up real-time data collection pipeline
6. 🔄 Implement continuous model retraining
7. 🔄 Add user authentication
8. 🔄 Set up monitoring and logging

## 💡 Tips for Mobile App Development

1. **Use the best model** from evaluation results
2. **Cache predictions** to reduce API calls
3. **Implement offline mode** with cached data
4. **Use WebSockets** for real-time alerts
5. **Add push notifications** for critical alerts
6. **Visualize on maps** using Google Maps/Mapbox
7. **Use charts library** (Chart.js, Victory Native)

## 🎉 You're Ready!

Run the training pipeline and start building your EpiWatch mobile application!

```powershell
# Start training
python src/models/train_all.py

# Start API (in another terminal)
uvicorn src.api.main:app --reload

# Check results
ls outputs/
```

Good luck with your project! 🚀🌍
