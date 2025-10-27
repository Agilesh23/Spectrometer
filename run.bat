@echo off
setlocal

REM Upgrade pip (optional)
python -m pip install --upgrade pip

REM Install dependencies for current user (no venv, no admin required)
python -m pip install --user -r requirements.txt

REM Optional: reduce telemetry
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

REM Run the app
python -m streamlit run app.py

endlocal
