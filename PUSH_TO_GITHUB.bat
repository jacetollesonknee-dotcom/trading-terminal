@echo off
echo.
echo ================================================
echo   AI Trading Terminal - Push to GitHub
echo ================================================
echo.

cd /d "%~dp0"
echo Current folder: %CD%
echo.

echo Checking git installation...
git --version
if errorlevel 1 (
    echo.
    echo ERROR: Git is not installed!
    echo Please download it from: https://git-scm.com/download/win
    echo Install it, then run this file again.
    pause
    exit
)

echo.
echo Setting up git repo...
git init
echo .env > .gitignore
echo schwab_token.json >> .gitignore
echo __pycache__/ >> .gitignore
echo *.pyc >> .gitignore

git add .
git commit -m "AI Trading Terminal - full build"
git branch -M main
git remote remove origin 2>nul
git remote add origin https://github.com/jacetollesonknee-dotcom/trading-terminal.git

echo.
echo ================================================
echo   About to push to GitHub...
echo   When asked for Username: jacetollesonknee-dotcom
echo   When asked for Password: paste your token
echo ================================================
echo.

git push -u origin main

echo.
if errorlevel 1 (
    echo PUSH FAILED - see error above
) else (
    echo SUCCESS! Code is now on GitHub.
)
echo.
pause
