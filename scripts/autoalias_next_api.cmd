@echo off
setlocal
set ROOT=%~dp0..
if not "%AUTOALIAS_PYTHON%"=="" (
  set PYTHON_EXE=%AUTOALIAS_PYTHON%
) else if exist "F:\ComfyUI\.venv\Scripts\python.exe" (
  set PYTHON_EXE=F:\ComfyUI\.venv\Scripts\python.exe
) else (
  set PYTHON_EXE=python
)
set PYTHONPATH=%ROOT%\src
if "%AUTOALIAS_WEB_OUT%"=="" set AUTOALIAS_WEB_OUT=%ROOT%\lan_reviews_next
"%PYTHON_EXE%" -m uvicorn autoalias.web_next.api:app --host 0.0.0.0 --port 8790 %*
exit /b %ERRORLEVEL%

