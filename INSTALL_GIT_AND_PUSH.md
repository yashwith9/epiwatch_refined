# 🔧 Install Git and Push to GitHub - Complete Guide

## 📥 **Step 1: Install Git**

### **Download Git:**
1. Go to: https://git-scm.com/download/windows
2. Download "64-bit Git for Windows Setup"
3. Run the installer with **default settings**
4. **Restart your command prompt** after installation

### **Verify Installation:**
```bash
git --version
```
You should see something like: `git version 2.42.0.windows.1`

---

## ⚙️ **Step 2: Configure Git (First Time Only)**

```bash
git config --global user.name "yashwith9"
git config --global user.email "your.email@example.com"
```

---

## 🚀 **Step 3: Push to GitHub**

### **Run these commands one by one:**

```bash
# 1. Initialize Git repository
git init

# 2. Add your GitHub repository as remote
git remote add origin https://github.com/yashwith9/epiwatch_wmad_refined.git

# 3. Add all files to staging
git add .

# 4. Create commit with message
git commit -m "🏥 EpiWatch: Complete AI epidemic detection system - 5 models, FastAPI, mobile integration"

# 5. Push to GitHub
git push -u origin main
```

### **If Step 5 fails, try:**
```bash
git push -u origin master
```

### **If you get authentication errors:**
```bash
# Use personal access token instead of password
# Go to GitHub → Settings → Developer settings → Personal access tokens
# Generate new token and use it as password
```

---

## 🔄 **Alternative: Quick Install and Push Script**

Save this as `quick_push.bat` and run it:

```batch
@echo off
echo 🏥 EpiWatch - GitHub Push Script
echo ================================

echo 📁 Initializing Git repository...
git init

echo 🔗 Adding remote repository...
git remote remove origin 2>nul
git remote add origin https://github.com/yashwith9/epiwatch_wmad_refined.git

echo 📋 Adding all files...
git add .

echo 💾 Creating commit...
git commit -m "🏥 EpiWatch: Complete AI epidemic detection system with 5 models, FastAPI, and mobile integration"

echo 🚀 Pushing to GitHub...
git push -u origin main

if %errorlevel% neq 0 (
    echo 🔄 Trying master branch...
    git push -u origin master
)

echo ✅ Done! Check your repository at:
echo https://github.com/yashwith9/epiwatch_wmad_refined
pause
```

---

## 🛠️ **Troubleshooting**

### **Problem: "git is not recognized"**
**Solution:** Install Git from https://git-scm.com/download/windows and restart command prompt

### **Problem: "Authentication failed"**
**Solutions:**
1. Use GitHub Desktop instead
2. Generate Personal Access Token on GitHub
3. Use SSH key authentication

### **Problem: "Repository not found"**
**Solution:** Make sure repository exists at https://github.com/yashwith9/epiwatch_wmad_refined

### **Problem: "Permission denied"**
**Solution:** Make sure you're logged into the correct GitHub account

---

## 📱 **Alternative: GitHub Desktop (Easiest)**

1. **Download GitHub Desktop:** https://desktop.github.com/
2. **Install and login** to your GitHub account
3. **Clone your repository** or **add existing repository**
4. **Commit and push** using the GUI

---

## ✅ **After Successful Push**

Your repository will contain:
- ✅ Complete AI epidemic detection system
- ✅ 5 trained models with performance comparison
- ✅ FastAPI server ready to run
- ✅ Mobile app integration guides
- ✅ Professional documentation
- ✅ Testing scripts

**Check your repository:** https://github.com/yashwith9/epiwatch_wmad_refined

---

## 🎯 **Quick Summary**

1. **Install Git** from https://git-scm.com/download/windows
2. **Restart command prompt**
3. **Run the 5 git commands** above
4. **Check your GitHub repository**

**Your EpiWatch AI system will be live on GitHub! 🌟**