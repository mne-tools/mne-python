#!/usr/bin/env python
"""
Test the EEG analysis tools with MNE sample data

This script downloads a sample EDF file and runs the analysis on it.
Perfect for testing without using real patient data!
"""

import os
import mne
from mne.datasets import testing

print("\n" + "="*60)
print("Testing EEG Analysis with Sample Data")
print("="*60 + "\n")

# Step 1: Download sample EDF data
print("Step 1: Downloading sample EDF file...")
print("(This only happens once, it will be cached)\n")

data_path = testing.data_path()
edf_path = data_path / 'EDF' / 'test_reduced.edf'

print(f"✓ Sample EDF file location: {edf_path}\n")

# Step 2: Test loading the file
print("Step 2: Testing file loading...")
raw = mne.io.read_raw_edf(edf_path, preload=True)
print(f"✓ Loaded successfully!")
print(f"  - Channels: {len(raw.ch_names)}")
print(f"  - Duration: {raw.times[-1]:.2f} seconds")
print(f"  - Sampling rate: {raw.info['sfreq']} Hz")
print(f"  - Channel names: {', '.join(raw.ch_names[:5])}...\n")

# Step 3: Run the full analysis
print("Step 3: Running full analysis with client_eeg_analysis.py...")
print("="*60 + "\n")

import subprocess
result = subprocess.run([
    'python', 'client_eeg_analysis.py',
    '--input', str(edf_path),
    '--output', 'sample_analysis_report.html'
], capture_output=False)

if result.returncode == 0:
    print("\n" + "="*60)
    print("TEST SUCCESSFUL!")
    print("="*60)
    print("\n✓ The tools are working correctly!")
    print("✓ Open 'sample_analysis_report.html' in your browser to see the results")
    print("\nNow you can use your own EDF files:")
    print("  python client_eeg_analysis.py --input YOUR_FILE.edf")
else:
    print("\n⚠ There was an issue. Please check the output above.")
