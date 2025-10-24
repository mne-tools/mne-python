#!/usr/bin/env python
"""
Simple Example: EEG Analysis with MNE-Python

This is a simplified example showing the basics of EEG analysis.
Perfect for getting started!

Usage:
    python simple_example.py
    (Edit the 'edf_file' variable below with your file path)
"""

import mne
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION - Edit this section for your data
# ============================================================================

# Path to your EDF file
edf_file = 'your_patient_data.edf'  # CHANGE THIS to your file path

# Output HTML report path
report_file = 'simple_report.html'

# Filter settings
low_freq = 0.5   # Low-pass filter (Hz)
high_freq = 40.0  # High-pass filter (Hz)
notch_freq = 50.0  # Line noise (50 Hz for Europe/Asia, 60 Hz for US)

# ============================================================================
# ANALYSIS - No need to edit below this line
# ============================================================================

print("\n" + "="*60)
print("EEG Analysis with MNE-Python")
print("="*60 + "\n")

# Step 1: Load the EDF file
print("Step 1: Loading EDF file...")
raw = mne.io.read_raw_edf(edf_file, preload=True)
print(f"   ✓ Loaded {len(raw.ch_names)} channels")
print(f"   ✓ Duration: {raw.times[-1]:.1f} seconds")
print(f"   ✓ Sampling rate: {raw.info['sfreq']} Hz\n")

# Step 2: Preprocess the data
print("Step 2: Preprocessing...")
print(f"   - Band-pass filter: {low_freq}-{high_freq} Hz")
raw.filter(l_freq=low_freq, h_freq=high_freq)
print(f"   - Notch filter: {notch_freq} Hz (line noise)")
raw.notch_filter(freqs=notch_freq)
print("   - Setting average reference")
raw.set_eeg_reference('average', projection=False)
print("   ✓ Preprocessing complete\n")

# Step 3: Create visualizations
print("Step 3: Creating visualizations...")

# 3a. Plot power spectral density
print("   - Power spectral density...")
fig_psd = raw.compute_psd(fmax=50).plot(average=True, show=False)
plt.savefig('psd_plot.png', dpi=150, bbox_inches='tight')
plt.close()

# 3b. Plot time series (first 10 seconds)
print("   - Time series plot...")
fig_ts = raw.plot(duration=10, n_channels=len(raw.ch_names), show=False)
plt.savefig('timeseries_plot.png', dpi=150, bbox_inches='tight')
plt.close()

print("   ✓ Plots saved: psd_plot.png, timeseries_plot.png\n")

# Step 4: Generate HTML report
print("Step 4: Generating HTML report...")
report = mne.Report(title='EEG Analysis Report')

# Add data overview
report.add_raw(raw=raw, title='EEG Data Overview', psd=True)

# Add basic info
info_html = f"""
<h2>Recording Information</h2>
<table>
    <tr><td><b>File:</b></td><td>{edf_file}</td></tr>
    <tr><td><b>Duration:</b></td><td>{raw.times[-1]:.2f} seconds</td></tr>
    <tr><td><b>Channels:</b></td><td>{len(raw.ch_names)}</td></tr>
    <tr><td><b>Sampling Rate:</b></td><td>{raw.info['sfreq']} Hz</td></tr>
</table>

<h3>Channel Names</h3>
<p>{', '.join(raw.ch_names)}</p>
"""
report.add_html(title='Information', html=info_html)

# Save report
report.save(report_file, overwrite=True, open_browser=False)
print(f"   ✓ Report saved: {report_file}\n")

# Step 5: Summary
print("="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print(f"\nGenerated files:")
print(f"  1. {report_file} (Open this in your browser)")
print(f"  2. psd_plot.png")
print(f"  3. timeseries_plot.png")
print(f"\nNext steps:")
print(f"  - Open {report_file} in your web browser")
print(f"  - For more features, use: python client_eeg_analysis.py --input {edf_file}")
print()
