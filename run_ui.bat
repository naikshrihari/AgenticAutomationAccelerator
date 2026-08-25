@echo off
REM ---------------------------------------------------------------------------
REM Double-click this file (or run it from cmd) to launch the AgentProbe web UI.
REM It activates the virtual environment if present and starts Streamlit, which
REM opens the app in your browser. No command-prompt knowledge needed.
REM ---------------------------------------------------------------------------

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

echo Starting AgentProbe UI...
echo If your browser does not open automatically, go to http://localhost:8501
python -m streamlit run app.py

pause
