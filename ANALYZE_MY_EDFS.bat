@echo off
REM ========================================================================
REM ANALYZE YOUR EDFS FOLDER - READY TO RUN!
REM ========================================================================
REM
REM This file is PRE-CONFIGURED for your exact EDFS folder.
REM Just double-click to run!
REM
REM Your folder: C:\Users\yildi\OneDrive\Desktop\braindecode-master\EDFS
REM ========================================================================

echo.
echo ========================================================================
echo ANALYZING YOUR EEG DATA
echo ========================================================================
echo.
echo Your EDFS folder:
echo   C:\Users\yildi\OneDrive\Desktop\braindecode-master\EDFS
echo.
echo This will:
echo   - Process all EDF files in your folder
echo   - Generate topographic maps for each file
echo   - Create power spectral density plots
echo   - Generate HTML reports for each recording
echo.
echo Results will be saved to:
echo   C:\Users\yildi\OneDrive\Desktop\braindecode-master\EDFS\reports\
echo.

pause

echo.
echo Checking Python installation...
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

echo Python found! ✓
echo.

echo Checking required packages...
python -c "import mne, matplotlib, numpy, scipy, edfio" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Required packages not found. Installing now...
    echo This will take a few minutes...
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
    echo Packages installed successfully! ✓
    echo.
)

echo All packages found! ✓
echo.

echo ========================================================================
echo STARTING ANALYSIS...
echo ========================================================================
echo.

REM Run the analysis with your exact path
python batch_process_windows.py --input "C:\Users\yildi\OneDrive\Desktop\braindecode-master\EDFS" --notch 50

if errorlevel 1 (
    echo.
    echo ========================================================================
    echo ERROR: Analysis failed!
    echo ========================================================================
    echo.
    echo Please check:
    echo   1. The EDFS folder path is correct
    echo   2. There are .edf files in the folder
    echo   3. The files are not corrupted
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo ANALYSIS COMPLETE! SUCCESS!
echo ========================================================================
echo.
echo Your reports are ready!
echo.
echo Location:
echo   C:\Users\yildi\OneDrive\Desktop\braindecode-master\EDFS\reports\
echo.
echo What you got:
echo   ✓ HTML reports (open in any browser)
echo   ✓ Topographic brain maps (PNG images)
echo   ✓ Power spectral density plots
echo   ✓ All visualizations organized in reports\figures\
echo.
echo Next steps:
echo   1. Open the 'reports' folder
echo   2. Double-click any .html file
echo   3. View your EEG analysis in your browser!
echo.
echo For AS EC QEEG.edf, look for:
echo   - AS_EC_QEEG_report.html
echo   - Strong alpha waves (8-13 Hz) in the frequency plot
echo   - Posterior (back of head) dominance in topographic maps
echo.
pause
