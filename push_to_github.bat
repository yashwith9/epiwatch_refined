@echo off
echo 🏥 EPIWATCH - GITHUB PUSH SCRIPT
echo ================================
echo 📦 Pushing AI epidemic detection system to GitHub
echo 🔗 Repository: https://github.com/yashwith9/epiwatch_nlp.git
echo ================================
echo.

echo 📁 Initializing Git repository...
git init
if %errorlevel% neq 0 (
    echo ❌ Git not found! Please install Git first:
    echo 🔗 https://git-scm.com/download/windows
    pause
    exit /b 1
)

echo 🔗 Adding remote repository...
git remote remove origin 2>nul
git remote add origin https://github.com/yashwith9/epiwatch_nlp.git

echo 📋 Adding all files...
git add .

echo 💾 Creating commit...
git commit -m "🏥 EpiWatch: Complete AI epidemic detection system - 5 models (DistilBERT, MuRIL, mBERT, XLM-RoBERTa, Custom LSTM), FastAPI server, mobile integration guides, ultra-fast inference (5ms)"

echo 🚀 Pushing to GitHub (main branch)...
git push -u origin main

if %errorlevel% neq 0 (
    echo 🔄 Main branch failed, trying master branch...
    git push -u origin master
    
    if %errorlevel% neq 0 (
        echo ❌ Push failed! Possible issues:
        echo   1. Authentication required - enter GitHub username/password
        echo   2. Repository doesn't exist
        echo   3. Network connection issues
        echo.
        echo 💡 Try GitHub Desktop for easier authentication:
        echo 🔗 https://desktop.github.com/
        pause
        exit /b 1
    )
)

echo.
echo ================================
echo 🎉 SUCCESS! Files pushed to GitHub
echo ================================
echo 🔗 Repository: https://github.com/yashwith9/epiwatch_nlp.git
echo 📚 Documentation: Check the docs/ folder
echo 🚀 API Server: Run python api/start_api.py
echo 📱 Mobile Integration: See docs/MOBILE_API_CALLS.md
echo ================================
echo.
echo ✅ Your EpiWatch AI system is now live on GitHub!
echo 🌟 Professional repository with complete documentation
echo ⚡ Ultra-fast epidemic detection ready for mobile apps
echo.
pause