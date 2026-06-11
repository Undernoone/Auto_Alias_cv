@echo off
setlocal
set ROOT=%~dp0..
cd /d "%ROOT%\webapp"
npm run dev
exit /b %ERRORLEVEL%

