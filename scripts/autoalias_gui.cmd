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
"%PYTHON_EXE%" -m autoalias.gui.desktop_editor --out "%ROOT%\lan_reviews" %*
exit /b %ERRORLEVEL%

