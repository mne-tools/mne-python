#!/usr/bin/env python
"""
EEG Client Data Analysis Script
================================

This script analyzes EDF files from clients and generates comprehensive reports
with topographic maps, power spectral density plots, and more.

Usage:
    python client_eeg_analysis.py --input patient_data.edf --output report.html

Author: EEG Analysis Tool
Date: 2025-10-24
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import mne

# Set up plotting style
mne.viz.set_browser_backend('matplotlib')


def load_edf_file(edf_path):
    """
    Load EDF file and return raw data object.

    Parameters
    ----------
    edf_path : str
        Path to the EDF file

    Returns
    -------
    raw : mne.io.Raw
        Raw EEG data object
    """
    print(f"\n{'='*60}")
    print(f"Loading EDF file: {edf_path}")
    print(f"{'='*60}")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=True)

    print(f"\n✓ File loaded successfully!")
    print(f"  - Duration: {raw.times[-1]:.2f} seconds ({raw.times[-1]/60:.2f} minutes)")
    print(f"  - Channels: {len(raw.ch_names)}")
    print(f"  - Sampling rate: {raw.info['sfreq']} Hz")

    return raw


def preprocess_data(raw, l_freq=0.5, h_freq=40.0, notch_freq=50.0):
    """
    Apply basic preprocessing to EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    l_freq : float
        Low cutoff frequency for band-pass filter (default: 0.5 Hz)
    h_freq : float
        High cutoff frequency for band-pass filter (default: 40 Hz)
    notch_freq : float
        Frequency for notch filter to remove line noise (default: 50 Hz)
        Set to 60.0 for US/North America data

    Returns
    -------
    raw : mne.io.Raw
        Preprocessed raw data
    """
    print(f"\n{'='*60}")
    print("Preprocessing Data")
    print(f"{'='*60}")

    # Apply band-pass filter
    print(f"✓ Applying band-pass filter: {l_freq}-{h_freq} Hz")
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    # Apply notch filter to remove line noise
    print(f"✓ Applying notch filter at {notch_freq} Hz (line noise removal)")
    raw.notch_filter(freqs=notch_freq, verbose=False)

    # Set average reference
    print("✓ Setting average reference")
    raw.set_eeg_reference('average', projection=False, verbose=False)

    print("\n✓ Preprocessing complete!")

    return raw


def create_topographic_maps(raw, output_dir):
    """
    Create topographic maps showing spatial distribution of signals.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed raw data
    output_dir : str
        Directory to save figures

    Returns
    -------
    fig_paths : list
        List of saved figure paths
    """
    print(f"\n{'='*60}")
    print("Creating Topographic Maps")
    print(f"{'='*60}")

    fig_paths = []

    # Create time windows for topographic maps
    duration = raw.times[-1]
    time_points = np.linspace(0, duration, min(6, int(duration)))[:5]  # Max 5 time points

    # Get data for each time point and create topomap
    for i, time_point in enumerate(time_points):
        print(f"✓ Creating topomap at t={time_point:.2f}s")

        # Get data at this time point
        data, times = raw.get_data(return_times=True)
        time_idx = np.argmin(np.abs(times - time_point))

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Plot topographic map
        mne.viz.plot_topomap(
            data[:, time_idx],
            raw.info,
            axes=ax,
            show=False,
            cmap='RdBu_r',
            vlim=(None, None)
        )

        ax.set_title(f'Topographic Map at t={time_point:.2f}s', fontsize=14, fontweight='bold')

        fig_path = os.path.join(output_dir, f'topomap_t{i+1}.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        fig_paths.append(fig_path)

    print(f"\n✓ Created {len(fig_paths)} topographic maps")

    return fig_paths


def create_power_spectral_density(raw, output_dir):
    """
    Create power spectral density plot.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed raw data
    output_dir : str
        Directory to save figure

    Returns
    -------
    fig_path : str
        Path to saved figure
    """
    print(f"\n{'='*60}")
    print("Creating Power Spectral Density Plot")
    print(f"{'='*60}")

    fig = raw.compute_psd(fmax=50).plot(show=False, average=True)
    fig.set_size_inches(12, 6)
    fig.suptitle('Power Spectral Density', fontsize=16, fontweight='bold')

    fig_path = os.path.join(output_dir, 'psd_plot.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("✓ PSD plot created")

    return fig_path


def create_time_series_plot(raw, output_dir, duration=10.0):
    """
    Create time series plot of EEG channels.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed raw data
    output_dir : str
        Directory to save figure
    duration : float
        Duration of time series to plot (default: 10 seconds)

    Returns
    -------
    fig_path : str
        Path to saved figure
    """
    print(f"\n{'='*60}")
    print(f"Creating Time Series Plot ({duration}s)")
    print(f"{'='*60}")

    fig = raw.plot(duration=duration, n_channels=len(raw.ch_names),
                   scalings='auto', show=False)

    fig_path = os.path.join(output_dir, 'time_series.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("✓ Time series plot created")

    return fig_path


def generate_html_report(raw, edf_path, output_path, fig_dir):
    """
    Generate comprehensive HTML report with all analyses.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed raw data
    edf_path : str
        Path to original EDF file
    output_path : str
        Path for output HTML report
    fig_dir : str
        Directory containing generated figures

    Returns
    -------
    report : mne.Report
        Generated report object
    """
    print(f"\n{'='*60}")
    print("Generating HTML Report")
    print(f"{'='*60}")

    # Create report
    report = mne.Report(title=f'EEG Analysis Report: {os.path.basename(edf_path)}')

    # Add raw data overview
    print("✓ Adding raw data overview")
    report.add_raw(raw=raw, title='Raw EEG Data', psd=True, projs=False)

    # Add topographic maps
    topomap_files = [f for f in os.listdir(fig_dir) if f.startswith('topomap_')]
    if topomap_files:
        print(f"✓ Adding {len(topomap_files)} topographic maps")
        for topo_file in sorted(topomap_files):
            topo_path = os.path.join(fig_dir, topo_file)
            report.add_image(image=topo_path, title=f'Topographic Map: {topo_file}')

    # Add PSD plot
    psd_path = os.path.join(fig_dir, 'psd_plot.png')
    if os.path.exists(psd_path):
        print("✓ Adding power spectral density plot")
        report.add_image(image=psd_path, title='Power Spectral Density')

    # Add time series
    ts_path = os.path.join(fig_dir, 'time_series.png')
    if os.path.exists(ts_path):
        print("✓ Adding time series plot")
        report.add_image(image=ts_path, title='Time Series View')

    # Add data info
    info_html = f"""
    <h2>Recording Information</h2>
    <ul>
        <li><b>File:</b> {os.path.basename(edf_path)}</li>
        <li><b>Duration:</b> {raw.times[-1]:.2f} seconds ({raw.times[-1]/60:.2f} minutes)</li>
        <li><b>Number of channels:</b> {len(raw.ch_names)}</li>
        <li><b>Sampling rate:</b> {raw.info['sfreq']} Hz</li>
        <li><b>Channel names:</b> {', '.join(raw.ch_names[:10])}{'...' if len(raw.ch_names) > 10 else ''}</li>
    </ul>
    """
    report.add_html(title='Recording Information', html=info_html)

    # Save report
    report.save(output_path, overwrite=True, open_browser=False)

    print(f"\n✓ Report saved to: {output_path}")

    return report


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description='Analyze EDF files and generate comprehensive EEG reports with topographic maps'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input EDF file'
    )
    parser.add_argument(
        '--output', '-o',
        default='eeg_analysis_report.html',
        help='Path for output HTML report (default: eeg_analysis_report.html)'
    )
    parser.add_argument(
        '--lowpass', '-l',
        type=float,
        default=0.5,
        help='Low-pass filter frequency in Hz (default: 0.5)'
    )
    parser.add_argument(
        '--highpass', '-h',
        type=float,
        default=40.0,
        help='High-pass filter frequency in Hz (default: 40.0)'
    )
    parser.add_argument(
        '--notch',
        type=float,
        default=50.0,
        help='Notch filter frequency for line noise (default: 50 Hz, use 60 for US data)'
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        return 1

    # Create output directory for figures
    output_dir = os.path.dirname(args.output) or '.'
    fig_dir = os.path.join(output_dir, 'eeg_figures')
    os.makedirs(fig_dir, exist_ok=True)

    try:
        # Step 1: Load data
        raw = load_edf_file(args.input)

        # Step 2: Preprocess
        raw = preprocess_data(raw, l_freq=args.lowpass, h_freq=args.highpass,
                              notch_freq=args.notch)

        # Step 3: Create visualizations
        create_topographic_maps(raw, fig_dir)
        create_power_spectral_density(raw, fig_dir)
        create_time_series_plot(raw, fig_dir)

        # Step 4: Generate HTML report
        generate_html_report(raw, args.input, args.output, fig_dir)

        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        print(f"\n✓ Report: {args.output}")
        print(f"✓ Figures: {fig_dir}/")
        print(f"\nOpen '{args.output}' in your web browser to view the full report.")

        return 0

    except Exception as e:
        print(f"\n{'='*60}")
        print("ERROR OCCURRED")
        print(f"{'='*60}")
        print(f"\n{str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
