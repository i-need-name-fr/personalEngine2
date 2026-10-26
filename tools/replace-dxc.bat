@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0replace-dxc.ps1" %*
echo.
pause
