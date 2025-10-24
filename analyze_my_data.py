#!/usr/bin/env python
"""
ANALYZE YOUR EEG DATA - PRE-CONFIGURED FOR YOUR EDFS FOLDER

This script is ready to run! It's configured for your exact folder:
C:\\Users\\yildi\\OneDrive\\Desktop\\braindecode-master\\EDFS

Just run: python analyze_my_data.py

Author: EEG Analysis Toolkit
Date: 2025-10-24
"""

import os
import sys

# ========================================================================
# YOUR CONFIGURATION - Pre-configured for your EDFS folder
# ========================================================================

YOUR_EDFS_FOLDER = r"C:\Users\yildi\OneDrive\Desktop\braindecode-master\EDFS"
NOTCH_FREQUENCY = 50  # 50 Hz for Europe/Asia, 60 Hz for US/Canada

# ========================================================================
# DO NOT EDIT BELOW THIS LINE
# ========================================================================

print("\n" + "="*70)
print("EEG ANALYSIS - YOUR DATA")
print("="*70)
print(f"\nYour EDFS folder: {YOUR_EDFS_FOLDER}")
print(f"Notch filter: {NOTCH_FREQUENCY} Hz")
print("\nThis will analyze all EDF files in your folder and generate:")
print("  ✓ Topographic brain maps")
print("  ✓ Power spectral density plots")
print("  ✓ Professional HTML reports")
print("\n" + "="*70 + "\n")

# Check if batch_process_windows.py exists
if not os.path.exists('batch_process_windows.py'):
    print("ERROR: batch_process_windows.py not found!")
    print("\nMake sure you have downloaded:")
    print("  - batch_process_windows.py")
    print("  - analyze_my_data.py (this file)")
    print("\nAnd they are in the same folder.")
    sys.exit(1)

# Check if the EDFS folder exists
if not os.path.exists(YOUR_EDFS_FOLDER):
    print(f"ERROR: EDFS folder not found at:")
    print(f"  {YOUR_EDFS_FOLDER}")
    print("\nPlease check:")
    print("  1. The path is correct")
    print("  2. You have access to OneDrive")
    print("  3. The folder exists")
    print("\nIf the path is different, edit line 17 in this file.")
    sys.exit(1)

# Check for EDF files
import glob
edf_files = glob.glob(os.path.join(YOUR_EDFS_FOLDER, "*.edf"))
if not edf_files:
    print(f"ERROR: No EDF files found in:")
    print(f"  {YOUR_EDFS_FOLDER}")
    print("\nPlease check:")
    print("  1. The folder contains .edf files")
    print("  2. The files have .edf extension (not .EDF)")
    sys.exit(1)

print(f"Found {len(edf_files)} EDF file(s) to analyze:")
for edf in edf_files[:5]:  # Show first 5
    print(f"  - {os.path.basename(edf)}")
if len(edf_files) > 5:
    print(f"  ... and {len(edf_files) - 5} more")
print()

# Import and run the batch processor
try:
    print("Loading analysis tools...")
    import subprocess

    # Run batch_process_windows.py
    result = subprocess.run([
        sys.executable,
        'batch_process_windows.py',
        '--input', YOUR_EDFS_FOLDER,
        '--notch', str(NOTCH_FREQUENCY)
    ])

    if result.returncode == 0:
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE! SUCCESS!")
        print("="*70)
        print("\nYour reports are ready!")
        print(f"\nLocation: {YOUR_EDFS_FOLDER}\\reports\\")
        print("\nWhat you got:")
        print("  ✓ HTML reports (open in any browser)")
        print("  ✓ Topographic brain maps")
        print("  ✓ Power spectral density plots")
        print("\nNext steps:")
        print("  1. Open the 'reports' folder")
        print("  2. Double-click any .html file")
        print("  3. View your EEG analysis!")
        print("\nFor AS EC QEEG.edf, look for:")
        print("  - Strong alpha waves (8-13 Hz)")
        print("  - Posterior dominance in brain maps")
        print()
    else:
        print("\n" + "="*70)
        print("ERROR: Analysis failed!")
        print("="*70)
        print("\nPlease check the error messages above.")
        sys.exit(1)

except Exception as e:
    print(f"\nERROR: {e}")
    print("\nMake sure you have installed the required packages:")
    print("  pip install mne matplotlib numpy scipy edfio")
    sys.exit(1)
