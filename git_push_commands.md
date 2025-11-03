# 🚀 Git Commands to Push to GitHub

## 📋 **Step-by-Step Git Commands**

Copy and paste these commands one by one:

### **Step 1: Install Git (if not installed)**
```bash
# Download and install Git from: https://git-scm.com/download/windows
# After installation, restart your command prompt
```

### **Step 2: Configure Git (first time only)**
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### **Step 3: Initialize Repository**
```bash
git init
```

### **Step 4: Add Remote Repository**
```bash
git remote add origin https://github.com/yashwith9/epiwatch_wmad_refined.git
```

### **Step 5: Add All Files**
```bash
git add .
```

### **Step 6: Create Commit**
```bash
git commit -m "🏥 EpiWatch: Complete AI epidemic detection system with 5 models, FastAPI, and mobile integration"
```

### **Step 7: Push to GitHub**
```bash
git push -u origin main
```

If that fails, try:
```bash
git push -u origin master
```

---

## 🔧 **Alternative: Force Push (if repository has conflicts)**
```bash
git push -f origin main
```

---

## ✅ **Verify Upload**
After pushing, check: https://github.com/yashwith9/epiwatch_wmad_refined