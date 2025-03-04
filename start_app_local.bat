@echo off
echo Starting Gradio Chat Application Locally...

:: Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed. Please install Python first.
    exit /b 1
)

:: Check if Node.js is installed (required for GitHub Copilot Language Server)
where node >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Warning: Node.js is not installed. GitHub Copilot integration will not work.
    echo Please install Node.js to use GitHub Copilot features.
)

:: Run the application locally
python run_app_local.py