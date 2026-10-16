@echo off
title DrowsyAlert - Phat hien ngu gat
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

echo ======================================================
echo           DrowsyAlert - He thong giam sat lai xe
echo ======================================================
echo Dang khoi dong ung dung tren trinh duyet...

if exist venv\Scripts\python.exe (
    call venv\Scripts\activate.bat
    venv\Scripts\python.exe -m streamlit run app.py
) else (
    python -m streamlit run app.py
)

pause
