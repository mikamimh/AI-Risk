@echo off
setlocal

cd /d "%~dp0"

rem Windows default: single-threaded joblib to avoid named-pipe issues
rem under Streamlit. Override by running `set AI_RISK_N_JOBS=N` before
rem invoking this script.
if not defined AI_RISK_N_JOBS set AI_RISK_N_JOBS=1

python -m streamlit run app.py

endlocal
