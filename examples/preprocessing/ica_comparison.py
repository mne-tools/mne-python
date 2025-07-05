"""
.. _ex-ica-comp:

===========================================================
Compare the performance of different ICA algorithms in MNE
===========================================================

This example compares various ICA algorithms (FastICA, Picard, Infomax,
Extended Infomax) on the same raw MEG data. For each algorithm:

- The ICA fit time (speed) is shown
- All components (up to 20) are visualized
- The EOG-related component from each method is detected and compared
- Comparison on clean vs noisy data is done

Note: In typical preprocessing, only one ICA algorithm is used.
This example is for educational purposes.
"""

# authors : Ganasekhar Kalla <ganasekharkalla@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

from pathlib import Path
from time import time

import numpy as np

import mne
from mne.datasets import sample
from mne.preprocessing import ICA

print(__doc__)

# %%

# Read and preprocess the data. Preprocessing consists of:
#
# - MEG channel selection
# - 1-30 Hz band-pass filter

# Load sample dataset
data_path = Path(sample.data_path())
raw_file = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
raw = mne.io.read_raw_fif(raw_file, preload=True)
raw.pick_types(meg=True, eeg=False, eog=True)
raw.crop(0, 60)

# %%

# Copy for clean and noisy
raw_clean = raw.copy()
raw_noisy = raw_clean.copy()
noise = 1e-12 * np.random.randn(*raw_noisy._data.shape)
raw_noisy._data += noise

# Rejection thresholds
reject_clean = dict(mag=5e-12, grad=4000e-13)
reject_noisy = dict(mag=1e-11, grad=8000e-13)

# %%


# Run ICA
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

    # Updated code with broken long line
    title = (
        f"ICA decomposition using {method} on "
        f"{'noisy' if raw_input is raw_noisy else 'clean'} data\n"
        f"(took {fit_time:.1f}s)"
    )
    ica.plot_components(title=title)

    return ica, fit_time


# %%


# Run all ICA methods
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

        eog_inds, _ = ica.find_bads_eog(raw_input, threshold=3.0, verbose="ERROR")
        if eog_inds:
            eog_components[full_label] = eog_inds[0]
            print(f"{full_label}:Detected EOG comp at index {eog_inds[0]}")
        else:
            eog_components[full_label] = None
            print(f"{full_label}: No EOG component detected")

    return icas, fit_times, eog_components


# %%


# Run on both raw versions
icas_clean, times_clean, eog_clean = run_all_ica(raw_clean, "clean", reject_clean)
icas_noisy, times_noisy, eog_noisy = run_all_ica(raw_noisy, "noisy", reject_noisy)

# Combine results
icas = {**icas_clean, **icas_noisy}
times = {**times_clean, **times_noisy}
eog_comps = {**eog_clean, **eog_noisy}

# %%

# Clean EOG components for each algorithm (Column 1)
for method in ["fastica", "picard", "infomax", "infomax_extended"]:
    key = f"clean_{method}"
    comp = eog_comps.get(key)
    if comp is not None:
        icas[key].plot_components(
            picks=[comp], title=f"{key} - EOG Component (Clean Data)", show=True
        )

# %%

# Noisy EOG components for each algorithm (Column 2)
for method in ["fastica", "picard", "infomax", "infomax_extended"]:
    key = f"noisy_{method}"
    comp = eog_comps.get(key)
    if comp is not None:
        icas[key].plot_components(
            picks=[comp], title=f"{key} - EOG Component (Noisy Data)", show=True
        )
