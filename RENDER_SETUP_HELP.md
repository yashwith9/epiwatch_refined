# 🎯 Render Deployment - Step by Step

## Current Screen: "New Web Service" Configuration

Here's what you need to do:

---

## 1️⃣ **Name** ✅
```
epiwatch_wmad_refined
```
(Already filled - looks good!)

---

## 2️⃣ **Project** (Optional - skip for now)
Leave as "Select a project..." ✅

---

## 3️⃣ **Language** ✅
Already set to **Python 3** ✅
(Keep this - it's correct!)

---

## 4️⃣ **Branch** ✅
Already set to **main** ✅
(Perfect! This is your GitHub branch with render.yaml)

---

## 5️⃣ **Region** ✅
Currently: **Oregon (US West)**
- ✅ Keep this (it's fine for now)
- Alternative: Choose the region closest to your users

---

## 6️⃣ **Root Directory** (Leave empty)
```
(empty - leave blank)
```
- This will use the root of your repo where `main.py` is located

---

## 7️⃣ **Build Command** (Scroll down to find this)
You should see these fields after scrolling:

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 8️⃣ **Plan** (Free Tier)
Make sure you select **Free** ✅

---

## 📋 Checklist Before Clicking "Create Web Service"

- [ ] Name: `epiwatch_wmad_refined`
- [ ] Language: `Python 3`
- [ ] Branch: `main`
- [ ] Region: `Oregon (US West)` or your choice
- [ ] Build Command: `pip install -r requirements.txt`
- [ ] Start Command: `uvicorn main:app --host 0.0.0.0 --port 8000`
- [ ] Plan: `Free`
- [ ] Root Directory: (empty)

---

## 🚀 Final Step

Once everything is filled in, scroll down and click the blue **"Create Web Service"** button!

⏳ Then wait 2-3 minutes for deployment to complete...

---

## ✅ After Deployment

You'll see:
- A live URL like: `https://epiwatch-api.onrender.com`
- Status showing "Live" (green)
- Deployment logs showing success

Then share that URL with your friend! 🎉
