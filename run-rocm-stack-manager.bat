@echo off
setlocal

cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo uv was not found on PATH.
    echo Install uv, then run this launcher again.
    exit /b 1
)

uv run --locked --extra ui python -m rocm_stack_manager.ui.app %*
exit /b %errorlevel%
