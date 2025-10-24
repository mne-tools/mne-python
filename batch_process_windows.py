#!/usr/bin/env python
"""
Batch Process EDF Files - Windows Compatible

This script processes all EDF files in a directory and generates reports.
Works directly on Windows!

Usage:
    python batch_process_windows.py --input "C:\path\to\EDFS" --output "C:\path\to\reports"
"""

import os
import sys
import argparse
import glob
from pathlib import Path

# Import analysis functions
try:
    import mne
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    print("\n❌ ERROR: Required packages not installed!")
    print("\nPlease install them first:")
    print("  pip install mne matplotlib numpy scipy")
    print("\nThen run this script again.")
    sys.exit(1)


def load_edf_file(edf_path):
    """Load EDF file and return raw data object."""
    print(f"  Loading: {os.path.basename(edf_path)}")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    print(f"    ✓ {len(raw.ch_names)} channels, {raw.times[-1]:.1f}s duration")
    return raw


def preprocess_data(raw, l_freq=0.5, h_freq=40.0, notch_freq=50.0):
    """Apply basic preprocessing to EEG data."""
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    raw.notch_filter(freqs=notch_freq, verbose=False)
    raw.set_eeg_reference('average', projection=False, verbose=False)
    return raw


def create_topographic_maps(raw, output_dir, file_prefix):
    """Create topographic maps showing spatial distribution of signals."""
    fig_paths = []

    # Create time windows for topographic maps
    duration = raw.times[-1]
    time_points = np.linspace(0, duration, min(6, int(duration)))[:5]

    for i, time_point in enumerate(time_points):
        data, times = raw.get_data(return_times=True)
        time_idx = np.argmin(np.abs(times - time_point))

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        try:
            mne.viz.plot_topomap(
                data[:, time_idx],
                raw.info,
                axes=ax,
                show=False,
                cmap='RdBu_r',
                vlim=(None, None)
            )
            ax.set_title(f'Topographic Map at t={time_point:.2f}s', fontsize=14, fontweight='bold')

            fig_path = os.path.join(output_dir, f'{file_prefix}_topomap_t{i+1}.png')
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            fig_paths.append(fig_path)
        except Exception as e:
            print(f"    ⚠ Warning: Could not create topomap {i+1}: {e}")
        finally:
            plt.close(fig)

    return fig_paths


def create_power_spectral_density(raw, output_dir, file_prefix):
    """Create power spectral density plot."""
    try:
        fig = raw.compute_psd(fmax=50).plot(show=False, average=True)
        fig.set_size_inches(12, 6)
        fig.suptitle('Power Spectral Density', fontsize=16, fontweight='bold')

        fig_path = os.path.join(output_dir, f'{file_prefix}_psd.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return fig_path
    except Exception as e:
        print(f"    ⚠ Warning: Could not create PSD plot: {e}")
        return None


def generate_html_report(raw, edf_path, output_path, fig_dir, file_prefix):
    """Generate comprehensive HTML report."""
    report = mne.Report(title=f'EEG Analysis: {os.path.basename(edf_path)}')

    # Add raw data overview
    try:
        report.add_raw(raw=raw, title='Raw EEG Data', psd=True, projs=False)
    except Exception as e:
        print(f"    ⚠ Warning: Could not add raw data to report: {e}")

    # Add topographic maps
    topomap_files = glob.glob(os.path.join(fig_dir, f'{file_prefix}_topomap_*.png'))
    for topo_file in sorted(topomap_files):
        try:
            report.add_image(image=topo_file, title=f'Topographic Map: {os.path.basename(topo_file)}')
        except Exception as e:
            print(f"    ⚠ Warning: Could not add topomap: {e}")

    # Add PSD plot
    psd_path = os.path.join(fig_dir, f'{file_prefix}_psd.png')
    if os.path.exists(psd_path):
        try:
            report.add_image(image=psd_path, title='Power Spectral Density')
        except Exception as e:
            print(f"    ⚠ Warning: Could not add PSD: {e}")

    # Add data info
    info_html = f"""
    <h2>Recording Information</h2>
    <ul>
        <li><b>File:</b> {os.path.basename(edf_path)}</li>
        <li><b>Duration:</b> {raw.times[-1]:.2f} seconds ({raw.times[-1]/60:.2f} minutes)</li>
        <li><b>Number of channels:</b> {len(raw.ch_names)}</li>
        <li><b>Sampling rate:</b> {raw.info['sfreq']} Hz</li>
        <li><b>Channel names:</b> {', '.join(raw.ch_names)}</li>
    </ul>
    """
    report.add_html(title='Recording Information', html=info_html)

    # Save report
    report.save(output_path, overwrite=True, open_browser=False)


def process_single_file(edf_path, output_dir, fig_dir, notch_freq=50.0):
    """Process a single EDF file."""
    try:
        # Get file prefix for naming
        file_prefix = Path(edf_path).stem.replace(' ', '_')

        # Load and preprocess
        raw = load_edf_file(edf_path)
        raw = preprocess_data(raw, notch_freq=notch_freq)

        # Create visualizations
        print("    Creating topographic maps...")
        create_topographic_maps(raw, fig_dir, file_prefix)

        print("    Creating PSD plot...")
        create_power_spectral_density(raw, fig_dir, file_prefix)

        # Generate report
        print("    Generating HTML report...")
        report_path = os.path.join(output_dir, f'{file_prefix}_report.html')
        generate_html_report(raw, edf_path, report_path, fig_dir, file_prefix)

        print(f"    ✓ Report saved: {os.path.basename(report_path)}\n")
        return True

    except Exception as e:
        print(f"    ❌ ERROR processing {os.path.basename(edf_path)}: {e}\n")
        return False


def main():
    """Main batch processing function."""
    parser = argparse.ArgumentParser(
        description='Batch process EDF files and generate comprehensive EEG reports'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to directory containing EDF files (e.g., "C:\\Users\\yildi\\OneDrive\\Desktop\\braindecode-master\\EDFS")'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Path for output directory (default: creates "reports" folder in input directory)'
    )
    parser.add_argument(
        '--notch',
        type=float,
        default=50.0,
        help='Notch filter frequency for line noise (default: 50 Hz, use 60 for US data)'
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.input):
        print(f"\n❌ ERROR: Input directory not found: {args.input}")
        return 1

    # Find all EDF files
    edf_pattern = os.path.join(args.input, '*.edf')
    edf_files = glob.glob(edf_pattern)

    if not edf_files:
        print(f"\n❌ ERROR: No EDF files found in: {args.input}")
        print(f"    Pattern searched: {edf_pattern}")
        return 1

    print(f"\n{'='*70}")
    print(f"BATCH EEG ANALYSIS")
    print(f"{'='*70}")
    print(f"\nFound {len(edf_files)} EDF file(s) to process:")
    for edf in edf_files:
        print(f"  - {os.path.basename(edf)}")
    print()

    # Create output directories
    if args.output is None:
        output_dir = os.path.join(args.input, 'reports')
    else:
        output_dir = args.output

    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Figures directory: {fig_dir}\n")
    print(f"{'='*70}\n")

    # Process each file
    successful = 0
    failed = 0

    for i, edf_path in enumerate(edf_files, 1):
        print(f"[{i}/{len(edf_files)}] Processing: {os.path.basename(edf_path)}")

        if process_single_file(edf_path, output_dir, fig_dir, args.notch):
            successful += 1
        else:
            failed += 1

    # Summary
    print(f"{'='*70}")
    print(f"BATCH PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"\n✓ Successfully processed: {successful} file(s)")
    if failed > 0:
        print(f"❌ Failed: {failed} file(s)")
    print(f"\nReports saved to: {output_dir}")
    print(f"Figures saved to: {fig_dir}")
    print(f"\nOpen the HTML reports in your browser to view the results!")

    return 0


if __name__ == '__main__':
    exit(main())
