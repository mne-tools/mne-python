@echo off
REM ========================================================================
REM EEG Analysis - One-Click Batch Processor
REM ========================================================================
REM
REM This batch file makes it easy to analyze your EDF files.
REM Just double-click this file to run!
REM
REM SETUP INSTRUCTIONS:
REM 1. Edit the paths below to match your file locations
REM 2. Save this file
REM 3. Double-click to run
REM ========================================================================

echo.
echo ========================================================================
echo EEG ANALYSIS - BATCH PROCESSOR
echo ========================================================================
echo.

REM ========================================================================
REM CONFIGURATION - EDIT THESE PATHS FOR YOUR SYSTEM
REM ========================================================================

REM Path to your EDF files folder
set INPUT_FOLDER=C:\Users\yildi\OneDrive\Desktop\braindecode-master\EDFS

REM Path where you want reports saved (leave empty to save in INPUT_FOLDER\reports)
set OUTPUT_FOLDER=

REM Notch filter frequency (50 for Europe/Asia, 60 for US/Canada)
set NOTCH_FREQ=50

REM ========================================================================
REM DO NOT EDIT BELOW THIS LINE
REM ========================================================================

echo Input folder: %INPUT_FOLDER%
echo Notch frequency: %NOTCH_FREQ% Hz
echo.

REM Check if input folder exists
if not exist "%INPUT_FOLDER%" (
    echo ERROR: Input folder does not exist!
    echo Please check the path: %INPUT_FOLDER%
    echo.
    pause
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo.
    echo Please install Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo Checking for required packages...
python -c "import mne, matplotlib, numpy, scipy" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Required packages not found. Installing now...
    echo This may take a few minutes...
    echo.
    pip install mne matplotlib numpy scipy edfio
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install packages!
        echo Please run manually: pip install mne matplotlib numpy scipy edfio
        echo.
        pause
        exit /b 1
    )
    echo.
    echo Packages installed successfully!
    echo.
)

echo.
echo Starting EEG analysis...
echo ========================================================================
echo.

REM Run the analysis
if "%OUTPUT_FOLDER%"=="" (
    python batch_process_windows.py --input "%INPUT_FOLDER%" --notch %NOTCH_FREQ%
) else (
    python batch_process_windows.py --input "%INPUT_FOLDER%" --output "%OUTPUT_FOLDER%" --notch %NOTCH_FREQ%
)

if errorlevel 1 (
    echo.
    echo ========================================================================
    echo ERROR: Analysis failed!
    echo ========================================================================
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo ANALYSIS COMPLETE!
echo ========================================================================
echo.
echo Reports are saved in: %INPUT_FOLDER%\reports
echo Figures are saved in: %INPUT_FOLDER%\figures
echo.
echo Open the HTML files in your browser to view the results!
echo.
pause
