"""
.. _ex-ica-comp:

===========================================================
Compare the performance of different ICA algorithms in MNE
===========================================================

This example compares various ICA algorithms (FastICA, Picard, Infomax, Extended Infomax)
on the same raw MEG data. For each algorithm:

- The ICA fit time (speed) is shown
- All components (up to 20) are visualized
- The EOG-related component from each method is detected and compared side-by-side
- Comparison on clean vs noisy data is done

This helps demonstrate practical differences in speed, stability, and ICA component shape.

Note: In a typical preprocessing pipeline, you would not run all ICA algorithms — this is
purely for educational comparison. All algorithms are run with the same random seed
(`random_state=0`) to ensure consistent behavior.
"""
print(__doc__)

import mne
from mne.preprocessing import ICA
from mne.datasets import sample
import numpy as np
import matplotlib.pyplot as plt
from time import time
from pathlib import Path

# Load MNE sample dataset
data_path = Path(sample.data_path())
raw_file = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(raw_file, preload=True)
raw.pick_types(meg=True, eeg=False, eog=True)
raw.crop(0, 60)  # work on a small subset for speed

# Clean copy
raw_clean = raw.copy()

# Add synthetic noise
raw_noisy = raw_clean.copy()
noise = 1e-12 * np.random.randn(*raw_noisy._data.shape)
raw_noisy._data += noise

# Set rejection thresholds
reject_clean = dict(mag=5e-12, grad=4000e-13)
reject_noisy = dict(mag=1e-11, grad=8000e-13)

# Function to run ICA
def run_ica(raw_input, method, fit_params=None, reject=None):
    print(f"\nRunning ICA with: {method}")
    ica = ICA(
        n_components=20,
        method=method,
        fit_params=fit_params,
        max_iter="auto",
        random_state=0,
    )
    t0 = time()
    ica.fit(raw_input, reject=reject)
    fit_time = time() - t0
    print(f"Fitting ICA took {fit_time:.1f}s.")
    ica.plot_components(title=f"ICA decomposition using {method} on {'noisy' if raw_input is raw_noisy else 'clean'} data\n(took {fit_time:.1f}s)")
    return ica, fit_time

# Run multiple ICA methods
def run_all_ica(raw_input, label, reject):
    icas = {}
    fit_times = {}
    eog_components = {}

    for method, params in [
        ("fastica", None),
        ("picard", None),
        ("infomax", None),
        ("infomax", {"extended": True}),
    ]:
        name = f"{method}" if not params else f"{method}_extended"
        full_label = f"{label}_{name}"
        ica, t = run_ica(raw_input, method, params, reject)
        icas[full_label] = ica
        fit_times[full_label] = t

        # Detect EOG component
        eog_inds, _ = ica.find_bads_eog(raw_input, threshold=3.0)
        if eog_inds:
            eog_components[full_label] = eog_inds[0]
            print(f"{full_label}: Detected EOG component at index {eog_inds[0]}")
        else:
            eog_components[full_label] = None
            print(f"{full_label}: No EOG component detected")

    return icas, fit_times, eog_components

# Run ICA on clean and noisy data
icas_clean, times_clean, eog_clean = run_all_ica(raw_clean, "clean", reject_clean)
icas_noisy, times_noisy, eog_noisy = run_all_ica(raw_noisy, "noisy", reject_noisy)

# Combine for comparison
icas = {**icas_clean, **icas_noisy}
times = {**times_clean, **times_noisy}
eog_comps = {**eog_clean, **eog_noisy}

# --- Separate clean and noisy plots ---
clean_labels = [label for label in eog_comps if label.startswith("clean")]
noisy_labels = [label for label in eog_comps if label.startswith("noisy")]

# Plot for clean data
fig_clean, axs_clean = plt.subplots(1, len(clean_labels), figsize=(18, 4))
for ax, label in zip(axs_clean, clean_labels):
    ica = icas[label]
    eog_idx = eog_comps[label]
    if eog_idx is not None:
        ica.plot_components(picks=[eog_idx], axes=ax, show=False, title=label)
    else:
        ax.set_title(f"{label}\nNo EOG component found")
fig_clean.suptitle("EOG Component Comparison (Clean Data)", fontsize=16)
plt.tight_layout()
plt.show()

# Plot for noisy data
fig_noisy, axs_noisy = plt.subplots(1, len(noisy_labels), figsize=(18, 4))
for ax, label in zip(axs_noisy, noisy_labels):
    ica = icas[label]
    eog_idx = eog_comps[label]
    if eog_idx is not None:
        ica.plot_components(picks=[eog_idx], axes=ax, show=False, title=label)
    else:
        ax.set_title(f"{label}\nNo EOG component found")
fig_noisy.suptitle("EOG Component Comparison (Noisy Data)", fontsize=16)
plt.tight_layout()
plt.show()

# Print timing comparison
print("\n=== ICA Fit Times ===")
for name, t in times.items():
    print(f"{name:20s}: {t:.2f} seconds")
