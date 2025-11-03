# EpiWatch System Architecture

## Complete End-to-End System Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA COLLECTION                              │
├─────────────────────────────────────────────────────────────────────┤
│  • GDELT News Articles                                              │
│  • NewsAPI                                                           │
│  • Twitter/Social Media                                             │
│  • Health Organization Reports                                      │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│  1. Language Detection (fastText)                                   │
│  2. Text Cleaning (URLs, special chars, HTML)                       │
│  3. Translation (MarianMT) → English                                │
│  4. Tokenization & Normalization                                    │
│  5. Stopword Removal (preserve epidemic keywords)                   │
│  6. Lemmatization                                                   │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       MODEL TRAINING                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  Model 1: Custom Neural Network (FROM SCRATCH)              │   │
│  │  ────────────────────────────────────────────────────────   │   │
│  │  • Embedding Layer (256 dim)                                │   │
│  │  • Bi-LSTM (128 hidden, 2 layers)                          │   │
│  │  • Attention Mechanism                                      │   │
│  │  • Dense Layers (64 → 32 → 1)                             │   │
│  │  • Sigmoid Output                                          │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  Model 2: XLM-RoBERTa (MULTILINGUAL)                       │   │
│  │  • 550 MB, 270M parameters                                  │   │
│  │  • Best for cross-lingual tasks                            │   │
│  │  • Fine-tuned for binary classification                    │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  Model 3: mBERT (104 LANGUAGES)                            │   │
│  │  • 680 MB, 179M parameters                                  │   │
│  │  • Multilingual BERT                                       │   │
│  │  • Fine-tuned for outbreak detection                       │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  Model 4: DistilBERT (FAST & EFFICIENT)                    │   │
│  │  • 270 MB, 135M parameters                                  │   │
│  │  • 60% faster than BERT                                    │   │
│  │  • Best for production deployment                          │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  Model 5: MuRIL (INDIAN LANGUAGES)                         │   │
│  │  • 890 MB, 237M parameters                                  │   │
│  │  • Specialized for Indian subcontinent                     │   │
│  │  • Supports Hindi, Bengali, Tamil, etc.                    │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MODEL EVALUATION & COMPARISON                     │
├─────────────────────────────────────────────────────────────────────┤
│  Metrics:                                                            │
│  • Accuracy, Precision, Recall, F1-Score                           │
│  • AUC-ROC, Confusion Matrix                                       │
│  • Inference Time (ms)                                             │
│  • Model Size (MB)                                                 │
│  • Weighted Performance Score                                      │
│                                                                      │
│  Output:                                                            │
│  • Comparison visualizations                                       │
│  • Best model recommendation                                       │
│  • Detailed metrics JSON                                           │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  REAL-TIME PREDICTION SYSTEM                         │
├─────────────────────────────────────────────────────────────────────┤
│  Input: News article, social media post, health report             │
│  Process: Preprocess → Best Model → Classification                 │
│  Output: [Outbreak: Yes/No] + Probability                          │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│               ANOMALY DETECTION & ALERT SYSTEM                       │
├─────────────────────────────────────────────────────────────────────┤
│  1. Aggregate predictions by:                                       │
│     • Region (Mumbai, Delhi, Nairobi, etc.)                        │
│     • Disease (Dengue, COVID-19, Malaria, etc.)                    │
│     • Time (hourly/daily)                                          │
│                                                                      │
│  2. Detect anomalies using:                                         │
│     • Z-Score Method (statistical)                                 │
│     • Isolation Forest (ML-based)                                  │
│     • Moving Average (time-series)                                 │
│                                                                      │
│  3. Calculate risk level:                                           │
│     • HIGH: Signal count × Probability ≥ 100                       │
│     • MODERATE: 50 ≤ Score < 100                                  │
│     • LOW: Score < 50                                             │
│                                                                      │
│  4. Generate alerts with:                                           │
│     • Location, disease, case count                                │
│     • Risk level and color coding                                  │
│     • Human-readable message                                       │
│     • Timestamp                                                    │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND SERVER                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Endpoints:                                                          │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ GET  /alerts          → Current outbreak alerts             │   │
│  │ GET  /trends?days=7   → 7-day disease trends               │   │
│  │ GET  /map             → Geospatial outbreak data           │   │
│  │ POST /detect          → Real-time text classification      │   │
│  │ GET  /stats           → System statistics                  │   │
│  │ GET  /health          → API health check                   │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Database: MongoDB (stores alerts, predictions, history)           │
│  Cache: Redis (fast access to recent data)                         │
│  Authentication: JWT tokens                                         │
└───────────────────────┬─────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MOBILE APPLICATION (FLUTTER)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  Screen 1: MAIN DASHBOARD                                   │   │
│  │  ─────────────────────────────────────────────────────────  │   │
│  │  • Interactive Map (Mapbox/Google Maps)                     │   │
│  │    - Color-coded regions (Red/Orange/Green)                │   │
│  │    - Hotspot markers                                       │   │
│  │  • Summary Cards:                                          │   │
│  │    - Total Cases: 8,081                                    │   │
│  │    - Countries: 8                                          │   │
│  │    - Critical Alerts: 2                                    │   │
│  │    - Regions Monitored: 6                                  │   │
│  │                                                            │   │
│  │  Data Source: GET /map + GET /stats                        │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  Screen 2: RECENT ALERTS FEED                              │   │
│  │  ─────────────────────────────────────────────────────────  │   │
│  │  • Scrollable list of alerts                               │   │
│  │  • Each alert card shows:                                  │   │
│  │    - Disease name + icon                                   │   │
│  │    - Location (city, country)                              │   │
│  │    - Risk badge (HIGH/MODERATE/LOW)                        │   │
│  │    - Case count                                            │   │
│  │    - Timestamp (e.g., "47 minutes ago")                    │   │
│  │    - Summary text                                          │   │
│  │  • Filter by risk level                                    │   │
│  │  • "Investigate Alert" button                              │   │
│  │                                                            │   │
│  │  Data Source: GET /alerts?limit=20                         │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  Screen 3: 7-DAY DISEASE TRENDS                            │   │
│  │  ─────────────────────────────────────────────────────────  │   │
│  │  • Bar charts for each disease                             │   │
│  │  • Diseases tracked:                                       │   │
│  │    - Dengue: 287 cases ↑                                  │   │
│  │    - Malaria: 134 cases →                                 │   │
│  │    - Cholera: 67 cases ↑                                  │   │
│  │    - Yellow Fever: 73 cases ↓                             │   │
│  │    - Measles: 68 cases →                                  │   │
│  │    - Influenza: 103 cases ↑                               │   │
│  │  • Trend indicators (↑↓→)                                  │   │
│  │  • Line chart toggle option                                │   │
│  │                                                            │   │
│  │  Data Source: GET /trends?days=7                           │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Additional Features:                                               │
│  • Push notifications for critical alerts                           │
│  • Offline mode with cached data                                    │
│  • Real-time updates via WebSocket                                  │
│  • Search and filter functionality                                  │
│  • User preferences and settings                                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Backend
- **Python 3.8+**
- **PyTorch** - Deep learning framework
- **Transformers** - Hugging Face pre-trained models
- **FastAPI** - REST API framework
- **MongoDB** - Database
- **Redis** - Caching

### NLP & ML
- **NLTK** - Text preprocessing
- **langdetect** - Language detection
- **scikit-learn** - ML utilities
- **statsmodels** - Time-series analysis

### Visualization
- **Matplotlib** - Static plots
- **Seaborn** - Statistical visualizations
- **Plotly** - Interactive charts

### Mobile App
- **Flutter/React Native** - Cross-platform
- **Mapbox/Google Maps** - Geospatial visualization
- **Chart.js/Victory Native** - Data visualization

## Data Flow for Mobile App

```
Mobile App Request
      │
      ▼
FastAPI Endpoint
      │
      ▼
Load Best Model
      │
      ▼
Make Prediction
      │
      ▼
Aggregate Results
      │
      ▼
Detect Anomalies
      │
      ▼
Generate Alerts
      │
      ▼
Format for Mobile
      │
      ▼
JSON Response
      │
      ▼
Mobile App Display
```

## Deployment Architecture

```
┌─────────────────┐
│  Mobile App     │
│  (Flutter)      │
└────────┬────────┘
         │
         │ HTTPS
         ▼
┌─────────────────┐     ┌──────────────┐
│  Load Balancer  │────▶│  CloudFlare  │
│  (Nginx)        │     │  (CDN/WAF)   │
└────────┬────────┘     └──────────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Server │
│  (Gunicorn)     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│MongoDB │ │ Redis  │
│Database│ │ Cache  │
└────────┘ └────────┘
```

## UN SDG Impact Mapping

```
┌─────────────────────────────────────────────────────────┐
│  SDG 3: Good Health & Well-being                        │
├─────────────────────────────────────────────────────────┤
│  ✓ Early outbreak detection                             │
│  ✓ Faster response time                                 │
│  ✓ Reduced morbidity & mortality                        │
│  ✓ Proactive health monitoring                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  SDG 9: Industry, Innovation & Infrastructure           │
├─────────────────────────────────────────────────────────┤
│  ✓ AI-powered health infrastructure                     │
│  ✓ Scalable technology solution                         │
│  ✓ Innovation in public health                          │
│  ✓ Digital transformation                               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  SDG 10: Reduced Inequalities                           │
├─────────────────────────────────────────────────────────┤
│  ✓ Focus on low-resource regions                        │
│  ✓ Multilingual support                                 │
│  ✓ Accessible technology                                │
│  ✓ Equal access to health information                   │
└─────────────────────────────────────────────────────────┘
```

---

**System Status:** ✅ Ready for Deployment
**Model Count:** 5 (1 Custom + 4 Pre-trained)
**API Endpoints:** 10
**Mobile Screens:** 3
**Languages Supported:** 100+
