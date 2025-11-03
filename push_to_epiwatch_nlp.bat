@echo off
echo 🏥 EPIWATCH NLP - GITHUB PUSH
echo ===============================
echo 📦 Pushing AI epidemic detection system
echo 🔗 Repository: https://github.com/yashwith9/epiwatch_nlp.git
echo ===============================
echo.

echo 📁 Initializing Git repository...
git init

echo 🔗 Adding remote repository...
git remote remove origin 2>nul
git remote add origin https://github.com/yashwith9/epiwatch_nlp.git

echo 📋 Adding all files...
git add .

echo 💾 Creating commit...
git commit -m "🏥 EpiWatch NLP: Complete AI epidemic detection system with 5 models, FastAPI server, and mobile integration"

echo 🚀 Pushing to GitHub...
git push -u origin main

if %errorlevel% neq 0 (
    echo 🔄 Trying master branch...
    git push -u origin master
)

echo.
echo ================================
echo 🎉 SUCCESS! Files pushed to GitHub
echo ================================
echo 🔗 Repository: https://github.com/yashwith9/epiwatch_nlp.git
echo 📚 View your project online now!
echo ================================
pause