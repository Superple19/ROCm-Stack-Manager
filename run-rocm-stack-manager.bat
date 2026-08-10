@echo off
setlocal

cd /d "%~dp0"
set "PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo ROCm Stack Manager venv was not found:
    echo   %PYTHON%
    echo Create it with: python -m venv .venv
    echo Then install the UI with: .venv\Scripts\python.exe -m pip install --editable ".[ui]"
    exit /b 1
)

"%PYTHON%" -c "import PySide6" >nul 2>&1
if errorlevel 1 (
    echo PySide6 is not installed in the Manager venv.
    echo Install it with: .venv\Scripts\python.exe -m pip install --editable ".[ui]"
    exit /b 1
)

"%PYTHON%" -m rocm_stack_manager.ui.app %*
exit /b %errorlevel%
