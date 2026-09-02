@echo off
REM ---------------------------------------------------------------------------
REM Double-click this file (or run it from cmd) to launch the Agentic Assurance Accelerator web UI.
REM It activates the virtual environment if present and starts Streamlit, which
REM opens the app in your browser. No command-prompt knowledge needed.
REM ---------------------------------------------------------------------------

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

echo Starting Agentic Assurance Accelerator UI...
echo Open the "Local URL" that Streamlit prints below (usually
echo http://localhost:8501, but it uses the next free port if that is taken).
echo Close any older Agentic Assurance Accelerator window first so you are not on a stale copy.
echo.
REM --server.port 8501 keeps the port stable; remove it to let Streamlit pick one.
python -m streamlit run app.py --server.port 8501

pause
