@echo off
setlocal
set ROOT=%~dp0..
cd /d "%ROOT%\webapp"
npm install
if errorlevel 1 exit /b %ERRORLEVEL%
npm run build
exit /b %ERRORLEVEL%

