@echo off
REM ========================================================================
REM AUTOMATIC SETUP - Installs Everything You Need!
REM ========================================================================
REM
REM This script does EVERYTHING for you:
REM   1. Checks if Python is installed
REM   2. Installs all required packages AUTOMATICALLY
REM   3. Verifies everything works
REM   4. Runs your first analysis!
REM
REM JUST DOUBLE-CLICK THIS FILE!
REM ========================================================================

color 0A
echo.
echo ========================================================================
echo           AUTOMATIC EEG ANALYSIS TOOLKIT SETUP
echo ========================================================================
echo.
echo This will:
echo   [1] Check Python installation
echo   [2] Install required packages AUTOMATICALLY
echo   [3] Verify everything works
echo   [4] Optionally run your first analysis
echo.
echo This is a ONE-TIME setup. After this, you can analyze EEG data anytime!
echo.
pause

echo.
echo ========================================================================
echo [STEP 1/4] Checking Python Installation...
echo ========================================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [X] ERROR: Python is NOT installed!
    echo.
    echo Please install Python first:
    echo   1. Go to: https://www.python.org/downloads/
    echo   2. Download Python 3.x
    echo   3. Run installer
    echo   4. CHECK THE BOX: "Add Python to PATH" ^(VERY IMPORTANT!^)
    echo   5. Click "Install Now"
    echo   6. Run this script again after installation
    echo.
    pause
    exit /b 1
)

python --version
echo [✓] Python is installed!
echo.
pause

echo.
echo ========================================================================
echo [STEP 2/4] Installing Required Packages AUTOMATICALLY...
echo ========================================================================
echo.
echo This will take 2-5 minutes depending on your internet speed.
echo You'll see lots of text scrolling - this is normal!
echo.
echo Installing: mne, matplotlib, numpy, scipy, edfio
echo.
pause

pip install --upgrade pip
pip install mne matplotlib numpy scipy edfio

if errorlevel 1 (
    echo.
    echo [X] ERROR: Package installation failed!
    echo.
    echo Trying alternative method...
    python -m pip install --upgrade pip
    python -m pip install mne matplotlib numpy scipy edfio

    if errorlevel 1 (
        echo.
        echo [X] ERROR: Installation still failed!
        echo.
        echo Please try manually:
        echo   python -m pip install mne matplotlib numpy scipy edfio
        echo.
        pause
        exit /b 1
    )
)

echo.
echo [✓] Packages installed successfully!
echo.
pause

echo.
echo ========================================================================
echo [STEP 3/4] Verifying Installation...
echo ========================================================================
echo.

python -c "import mne, matplotlib, numpy, scipy, edfio; print('[✓] MNE-Python: ' + mne.__version__); print('[✓] NumPy: ' + numpy.__version__); print('[✓] Matplotlib: ' + matplotlib.__version__); print('[✓] SciPy: ' + scipy.__version__); print('[✓] edfio: OK'); print(''); print('[✓✓✓] ALL PACKAGES WORKING!')"

if errorlevel 1 (
    echo [X] ERROR: Verification failed!
    echo.
    pause
    exit /b 1
)

echo.
pause

echo.
echo ========================================================================
echo [STEP 4/4] Setup Complete!
echo ========================================================================
echo.
echo [✓✓✓] EVERYTHING IS READY!
echo.
echo You can now analyze your EEG data!
echo.
echo Your EDF folder:
echo   C:\Users\yildi\OneDrive\Desktop\braindecode-master\EDFS
echo.
echo ========================================================================
echo.
echo What do you want to do now?
echo.
echo   [1] Analyze my EEG data NOW
echo   [2] Exit (I'll analyze later)
echo.
set /p choice="Enter your choice (1 or 2): "

if "%choice%"=="1" (
    echo.
    echo ========================================================================
    echo Starting Analysis...
    echo ========================================================================
    echo.

    REM Check if batch_process_windows.py exists
    if exist batch_process_windows.py (
        python batch_process_windows.py --input "C:\Users\yildi\OneDrive\Desktop\braindecode-master\EDFS" --notch 50

        if errorlevel 1 (
            echo.
            echo [X] Analysis encountered an error.
            echo Please check that your EDFS folder exists and contains .edf files.
            echo.
        ) else (
            echo.
            echo ========================================================================
            echo [✓✓✓] ANALYSIS COMPLETE!
            echo ========================================================================
            echo.
            echo Your reports are ready at:
            echo   C:\Users\yildi\OneDrive\Desktop\braindecode-master\EDFS\reports\
            echo.
            echo Double-click any .html file to view in your browser!
            echo.
        )
    ) else (
        echo.
        echo [!] File not found: batch_process_windows.py
        echo.
        echo Please make sure you downloaded:
        echo   - SETUP_EVERYTHING.bat ^(this file^)
        echo   - batch_process_windows.py
        echo.
        echo And they are in the same folder.
        echo.
    )
) else (
    echo.
    echo [✓] Setup complete!
    echo.
    echo When you're ready to analyze, just run:
    echo   - ANALYZE_MY_EDFS.bat ^(double-click it^)
    echo   - OR: python analyze_my_data.py
    echo.
)

echo.
echo ========================================================================
echo Thank you for using the EEG Analysis Toolkit!
echo ========================================================================
echo.
pause
