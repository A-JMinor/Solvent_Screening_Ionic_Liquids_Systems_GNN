@echo off

REM to run this file: ./run.bat

REM Add folder to PYTHONPATH
set PYTHONPATH=%CD%

REM Run Python script for solvent pre-selection
python src/01_screening.py --metric custom_inf
