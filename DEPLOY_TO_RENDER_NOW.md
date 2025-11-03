# 🚀 Deploy to Render in 3 Steps

## ✅ Your GitHub Repo is Ready
- URL: https://github.com/yashwith9/epiwatch_wmad_refined.git
- Branch: `main`
- Files pushed: ✅ `render.yaml` + `requirements.txt` + All code

---

## 🎯 Deploy to Render (Right Now)

### Step 1: Go to Render
Open: https://render.com

### Step 2: Sign Up / Login
- Click "Sign Up" or "Login"
- Choose "Sign up with GitHub" (easiest)
- Authorize Render to access your GitHub repos

### Step 3: Create Web Service
1. Click **"New +"** button (top right)
2. Click **"Web Service"**
3. Find and select: **`epiwatch_wmad_refined`** repo
4. Click **"Connect"**

### Step 4: Configure (Render auto-detects most settings)
- **Name**: `epiwatch-api` (or anything you want)
- **Region**: Choose closest to you
- **Branch**: `main` ✅
- **Runtime**: `Python 3.11` ✅
- **Build Command**: `pip install -r requirements.txt` ✅
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port 8000` ✅
- **Plan**: `Free` ✅

### Step 5: Deploy
Click **"Create Web Service"** button

⏳ Wait 2-3 minutes while Render deploys...

---

## 🎉 You're Done!

After deployment, Render gives you a public URL like:

```
https://epiwatch-api.onrender.com
```

**This is your API URL!** ✅

---

## 📱 Share with Your Friend

Your friend uses this URL in their mobile app:

```javascript
const API_BASE_URL = 'https://epiwatch-api.onrender.com';

// All endpoints work:
// GET https://epiwatch-api.onrender.com/alerts
// GET https://epiwatch-api.onrender.com/map/data
// GET https://epiwatch-api.onrender.com/trends
// GET https://epiwatch-api.onrender.com/dashboard/stats
// POST https://epiwatch-api.onrender.com/predict
```

---

## ✨ What Happens Next

1. **Render builds** your app (installs dependencies)
2. **Starts the API** (runs `uvicorn main:app`)
3. **Gives you a public URL** (https://epiwatch-api.onrender.com)
4. **Auto-updates** whenever you push to GitHub! 🔄

---

## 🧪 Test Your Deployed API

Once deployment is complete:

### In Browser
```
https://epiwatch-api.onrender.com/docs
```
You'll see the interactive Swagger documentation!

### With cURL
```bash
curl https://epiwatch-api.onrender.com/alerts
curl https://epiwatch-api.onrender.com/dashboard/stats
```

### With JavaScript
```javascript
const API_BASE_URL = 'https://epiwatch-api.onrender.com';

fetch(`${API_BASE_URL}/alerts`)
  .then(r => r.json())
  .then(data => console.log(data));
```

---

## ⚠️ Important Notes

- **Free tier**: Goes to sleep after 15 min of inactivity (takes 30 sec to wake up)
- **Upgrade**: $7/month to keep always-on
- **Auto-deploy**: Every time you `git push`, Render automatically redeploys!

---

## 🎯 Next Steps

1. ✅ Files pushed to GitHub
2. ✅ `render.yaml` created
3. ⏳ Go to Render and deploy
4. 📱 Share URL with your friend
5. 🚀 Friend integrates into mobile app

**That's it!** Your API is now live and shareable! 🎉
