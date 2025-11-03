# 🚀 Deploy EpiWatch API to Render

## Why Render?
✅ **Free tier** with generous limits  
✅ **Auto-deploys** from GitHub  
✅ **Public URL** instantly  
✅ **No credit card needed** (for free tier)  
✅ **Simple setup** - just 5 minutes  
✅ **Your friend can use immediately**

---

## 📋 Step-by-Step Deployment Guide

### **Step 1: Prepare Your Repository**

Make sure your GitHub repo has these files in the root directory:

```
NLP/
├── main.py                 (Your FastAPI app)
├── requirements.txt        (Python dependencies)
├── render.yaml            (Create this - see below)
├── src/
│   ├── models/
│   ├── preprocessing/
│   └── evaluation/
├── models/saved/
│   └── custom_best.pt
└── data/processed/
    └── epidemic_data.csv
```

### **Step 2: Create `requirements.txt`**

Make sure you have this file with all dependencies:

```bash
# Run this to generate requirements.txt
pip freeze > requirements.txt
```

Your `requirements.txt` should include:
```
fastapi==0.104.1
uvicorn==0.24.0
torch==2.0.0
transformers==4.35.0
scikit-learn==1.3.2
pandas==2.0.3
numpy==1.24.3
nltk==3.8.1
langdetect==1.0.9
```

### **Step 3: Create `render.yaml`**

Create a new file `render.yaml` in your project root:

```yaml
services:
  - type: web
    name: epiwatch-api
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port 8000
    envVars:
      - key: PYTHON_VERSION
        value: 3.11
```

### **Step 4: Push to GitHub**

```bash
git add .
git commit -m "Add Render deployment configuration"
git push origin main
```

### **Step 5: Deploy on Render**

1. **Go to** https://render.com
2. **Sign up** with GitHub (recommended)
3. **Click** "New +" → "Web Service"
4. **Connect** your GitHub repo
5. **Configure:**
   - Name: `epiwatch-api`
   - Runtime: `Python 3.11`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port 8000`
   - Plan: `Free`
6. **Click** "Create Web Service"
7. **Wait** 2-3 minutes for deployment
8. **Get your public URL** (shown in dashboard)

---

## 🎯 Your Public API URL

After deployment, you'll get a URL like:

```
https://epiwatch-api.onrender.com
```

**Share this with your friend!** ✅

---

## 📱 For Your Friend's Mobile App

Your friend should use:

```javascript
const API_BASE_URL = 'https://epiwatch-api.onrender.com';

// Then use all endpoints like:
// GET https://epiwatch-api.onrender.com/alerts
// GET https://epiwatch-api.onrender.com/map/data
// GET https://epiwatch-api.onrender.com/trends
// POST https://epiwatch-api.onrender.com/predict
```

---

## ⚙️ Important Settings for Render

### **Environment Variables** (if needed)

In Render Dashboard → Environment:

```
PYTHONUNBUFFERED=true
```

This ensures logs appear in real-time.

### **Auto-Deploy on GitHub Push**

Render automatically redeploys when you push to GitHub:

```bash
git push origin main
# → Render detects change → Auto-deploys within 1-2 minutes
```

---

## 🔍 Monitor Your API

After deployment:

1. **View Logs**: Dashboard → "Logs" tab
2. **Test Endpoint**: Visit https://epiwatch-api.onrender.com/docs
3. **Check Status**: Dashboard shows "Live" when running
4. **View Metrics**: See CPU, memory usage

---

## ⚠️ Free Tier Limitations

| Feature | Free Tier | Paid |
|---------|-----------|------|
| Uptime | 99.9% SLA | 99.9% SLA |
| Sleep | 15 min inactivity | Always on |
| Requests | Unlimited | Unlimited |
| Bandwidth | 100 GB/month | More |
| Cost | Free | $7/month+ |

**Note**: Free tier spins down after 15 minutes of inactivity (takes 30 seconds to wake up)

To upgrade to paid ($7/month) for always-on:
- Render Dashboard → Settings → Change Plan to "Starter"

---

## 🧪 Test Your Deployed API

### **Test in Browser**
```
https://epiwatch-api.onrender.com/docs
```

### **Test with cURL**
```bash
# Get alerts
curl https://epiwatch-api.onrender.com/alerts

# Get dashboard stats
curl https://epiwatch-api.onrender.com/dashboard/stats

# Get trends
curl https://epiwatch-api.onrender.com/trends
```

### **Test from JavaScript**
```javascript
const API_BASE_URL = 'https://epiwatch-api.onrender.com';

// Test alerts endpoint
fetch(`${API_BASE_URL}/alerts`)
  .then(r => r.json())
  .then(data => console.log('Alerts:', data));

// Test prediction endpoint
fetch(`${API_BASE_URL}/predict`, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    text: 'Sudden spike in dengue cases reported',
    location: 'Mumbai'
  })
})
  .then(r => r.json())
  .then(data => console.log('Prediction:', data));
```

---

## 🐛 Troubleshooting

### **Deployment Fails**
- Check logs: Dashboard → Logs
- Common issues:
  - Missing `requirements.txt`
  - Missing `render.yaml`
  - Syntax errors in `main.py`

### **API Returns 500 Error**
- Check logs for stack trace
- Ensure all data files are in repo:
  - `models/saved/custom_best.pt`
  - `data/processed/epidemic_data.csv`

### **Slow Response (First Request)**
- Free tier goes to sleep after 15 min
- First request wakes it up (takes 30 seconds)
- Solution: Upgrade to paid tier or use a "ping" service

### **CORS Issues with Mobile App**
Your `main.py` should have CORS enabled:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📲 Integration Examples for Mobile App

### **Vue.js / React**
```javascript
const API_BASE_URL = 'https://epiwatch-api.onrender.com';

// Alerts
async function getAlerts() {
  const response = await fetch(`${API_BASE_URL}/alerts`);
  return await response.json();
}

// Trends
async function getTrends() {
  const response = await fetch(`${API_BASE_URL}/trends`);
  return await response.json();
}

// Map data
async function getMapData() {
  const response = await fetch(`${API_BASE_URL}/map/data`);
  return await response.json();
}

// Make prediction
async function predict(text, location) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ text, location })
  });
  return await response.json();
}
```

### **React Native**
```javascript
const API_BASE_URL = 'https://epiwatch-api.onrender.com';

useEffect(() => {
  fetch(`${API_BASE_URL}/dashboard/stats`)
    .then(r => r.json())
    .then(data => setStats(data));
}, []);
```

---

## 🔐 Security Notes

1. **Your data is public** - Anyone with your Render URL can access it
   - Solution: Add API authentication if needed

2. **Models are included** - Training data and weights are in the repo
   - Solution: Use private GitHub repo if concerned

3. **Rate limiting** - Render is free but has fair-use limits
   - Solution: Upgrade to paid for production

---

## 📞 Support

- **Render Docs**: https://render.com/docs
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **GitHub Issues**: Post questions on your repo

---

## ✅ Deployment Checklist

- [ ] Created `requirements.txt` with all dependencies
- [ ] Created `render.yaml` in root directory
- [ ] Pushed code to GitHub
- [ ] Signed up on Render.com
- [ ] Connected GitHub repo to Render
- [ ] Deployed successfully
- [ ] Got public URL from Render
- [ ] Tested `/docs` endpoint
- [ ] Shared URL with friend
- [ ] Friend can access all endpoints

---

## 🎉 Done!

Your EpiWatch API is now **live on Render**!

**Share this URL with your friend:**
```
https://epiwatch-api.onrender.com
```

They can integrate it into their mobile app and start using all the endpoints! 🚀
