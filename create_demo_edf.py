#!/usr/bin/env python
"""
Create a demo EDF file with synthetic EEG data for testing

This creates a fake EEG recording that you can use to test the analysis tools.
"""

import numpy as np
from datetime import datetime

try:
    import mne
    from mne import create_info
    from mne.io import RawArray

    print("\n" + "="*60)
    print("Creating Demo EDF File")
    print("="*60 + "\n")

    # Set parameters
    n_channels = 10
    sfreq = 256  # Sampling frequency (Hz)
    duration = 60  # 60 seconds of data

    # Create channel names (standard 10-20 system)
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
    ch_types = ['eeg'] * n_channels

    # Create synthetic EEG data
    print("Generating synthetic EEG data...")
    n_samples = int(duration * sfreq)
    data = np.zeros((n_channels, n_samples))

    # Add different frequency components (alpha, beta, theta)
    t = np.arange(n_samples) / sfreq

    for i in range(n_channels):
        # Alpha waves (8-13 Hz) - dominant when relaxed
        alpha = 10 * np.sin(2 * np.pi * 10 * t + i * 0.5)

        # Beta waves (13-30 Hz) - active thinking
        beta = 5 * np.sin(2 * np.pi * 20 * t + i * 0.3)

        # Theta waves (4-8 Hz) - drowsiness
        theta = 8 * np.sin(2 * np.pi * 6 * t + i * 0.7)

        # Add some random noise
        noise = np.random.randn(n_samples) * 2

        # Combine all components
        data[i, :] = alpha + beta + theta + noise

    # Convert to microvolts
    data = data * 1e-6

    # Create MNE info structure
    print("Creating MNE Raw object...")
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = RawArray(data, info)

    # Add standard montage (electrode positions)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    # Save as EDF
    output_file = 'demo_eeg_data.edf'
    print(f"Saving to {output_file}...")
    raw.export(output_file, overwrite=True)

    print("\n✓ Demo EDF file created successfully!")
    print(f"\nFile details:")
    print(f"  - Filename: {output_file}")
    print(f"  - Channels: {n_channels}")
    print(f"  - Duration: {duration} seconds")
    print(f"  - Sampling rate: {sfreq} Hz")
    print(f"\nNow test the analysis:")
    print(f"  python client_eeg_analysis.py --input {output_file}")

except ImportError as e:
    print("\n⚠ MNE-Python is not installed yet.")
    print("\nTo install MNE and test:")
    print("  pip install mne")
    print("  python create_demo_edf.py")
    print("\nOR simply provide your own EDF file:")
    print("  python client_eeg_analysis.py --input YOUR_FILE.edf")
