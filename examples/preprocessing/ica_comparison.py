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

# Authors: Pierre Ablin <pierreablin@gmail.com>
#          Ganasekhar Kalla <ganasekharkalla@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

import warnings
from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning

import mne
from mne.datasets import sample
from mne.preprocessing import ICA

print(__doc__)

# Reduce console noise from MNE and sklearn
mne.set_log_level("ERROR")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# %%

# Read and preprocess the data. Preprocessing consists of:
#
# - MEG channel selection
# - 1-30 Hz band-pass filter

# Load sample dataset
data_path = sample.data_path()
raw_file = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
raw = mne.io.read_raw_fif(raw_file).crop(0, 60).pick(["meg", "eog"]).load_data()

# %%

# Copy for clean
raw_clean = raw


def _scale_to_rms(noise, target_rms):
    curr_rms = np.sqrt(np.mean(noise**2, axis=1, keepdims=True)) + 1e-30
    return noise * (target_rms / curr_rms)


# Noise generators
def _gaussian_noise(shape, rng):
    return rng.randn(*shape)


def _pink_noise(shape, rng, sfreq):
    n_channels, n_times = shape
    # Build frequency weights ~ 1/sqrt(f) to get 1/f power spectrum
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    weights = np.ones_like(freqs)
    nonzero = freqs > 0
    weights[nonzero] = 1.0 / np.sqrt(freqs[nonzero])
    noise = rng.randn(n_channels, n_times)
    noise_fft = np.fft.rfft(noise, axis=1)
    noise_fft *= weights[np.newaxis, :]
    pink = np.fft.irfft(noise_fft, n=n_times, axis=1)
    return pink


def _line_noise(shape, rng, sfreq, line_freq):
    n_channels, n_times = shape
    t = np.arange(n_times) / sfreq
    nyq = sfreq / 2.0
    harmonics = [h for h in [1, 2, 3] if h * line_freq < nyq]
    base = np.zeros((n_channels, n_times))
    for h in harmonics:
        phase = rng.rand(n_channels, 1) * 2 * np.pi
        amp = 1.0 / h
        base += amp * np.sin(2 * np.pi * h * line_freq * t + phase)
    return base


def _emg_bursts(
    shape, rng, sfreq, low=20.0, high=100.0, burst_prob=0.01, burst_len_s=0.2
):
    n_channels, n_times = shape
    # Start with band-limited noise in EMG band via FFT masking
    white = rng.randn(n_channels, n_times)
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    mask = (freqs >= low) & (freqs <= high)
    white_fft = np.fft.rfft(white, axis=1)
    white_fft[:, ~mask] = 0.0
    emg_band = np.fft.irfft(white_fft, n=n_times, axis=1)
    # Create sparse burst envelopes
    burst_len = max(1, int(burst_len_s * sfreq))
    envelope = np.zeros((n_channels, n_times))
    for ch in range(n_channels):
        idx = 0
        while idx < n_times:
            if rng.rand() < burst_prob:
                end = min(n_times, idx + burst_len)
                envelope[ch, idx:end] = 1.0
                idx = end
            else:
                idx += burst_len
    return emg_band * envelope


# Helper: add noise to reach target SNR (in dB) with selectable type
def add_noise_for_snr(
    raw_input, snr_db, random_state=0, noise_type="gaussian", line_freq=50
):
    rng = np.random.RandomState(random_state)
    data = raw_input._data
    sfreq = raw_input.info["sfreq"]
    # Per-channel RMS so SNR is matched channel-wise
    signal_rms = np.sqrt(np.mean(data**2, axis=1, keepdims=True)) + 1e-30
    amp_ratio = 10 ** (-snr_db / 20.0)
    noise_rms = amp_ratio * signal_rms

    if noise_type == "gaussian":
        noise = _gaussian_noise(data.shape, rng)
    elif noise_type == "pink":
        noise = _pink_noise(data.shape, rng, sfreq)
    elif noise_type in ("line50", "line60"):
        lf = 50 if noise_type == "line50" else 60
        noise = _line_noise(data.shape, rng, sfreq, lf)
    elif noise_type == "emg":
        noise = _emg_bursts(data.shape, rng, sfreq)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    noise = _scale_to_rms(noise, noise_rms)
    raw_noisy_local = raw_input.copy()
    raw_noisy_local._data = data + noise
    return raw_noisy_local, amp_ratio


# Baseline rejection thresholds for clean data
reject_clean = dict(mag=5e-12, grad=4000e-13)

# Choose SNR levels (in dB)
snr_levels = [10, 0]
# Choose noise types to evaluate: 'gaussian', 'pink', 'line50'/'line60', 'emg'
noise_types = ["gaussian", "pink", "line50", "emg"]

# %%


# Run ICA
def run_ica(
    raw_input, method, fit_params=None, reject=None, label=None, display_name=None
):
    name_for_print = display_name if display_name is not None else method
    print(f"\nRunning ICA with: {name_for_print}")
    ica = ICA(
        n_components=20,
        method=method,
        fit_params=fit_params,
        max_iter="auto",
        random_state=0,
    )
    # Emit informational lines similar to MNE's verbose output
    n_channels = raw_input.info["nchan"]
    print(
        f"Fitting ICA to data using {n_channels} channels"
        f"(please be patient, this may take a while)"
    )
    print("Selecting by number: 20 components")
    t0 = time()
    # Suppress verbose logs during fitting
    with mne.use_log_level("ERROR"):
        try:
            ica.fit(raw_input, reject=reject, verbose="ERROR")
        except RuntimeError as err:
            msg = str(err)
            if "No clean segment found" in msg:
                print(
                    "No clean segment with current reject; retrying without rejection …"
                )
                ica.fit(raw_input, reject=None, verbose="ERROR")
            else:
                raise
    fit_time = time() - t0
    print(f"Fitting ICA took {fit_time:.1f}s.")

    data_label = label if label is not None else "data"
    title = (
        f"ICA decomposition using {name_for_print} on {data_label}\n"
        f"(took {fit_time:.1f}s)"
    )
    ica.plot_components(title=title)
    plt.close()

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
        # Clarify label and display name for extended infomax
        is_extended = method == "infomax" and params and params.get("extended", False)
        name = "infomax_extended" if is_extended else method
        display_name = "infomax (extended)" if is_extended else method
        full_label = f"{label}_{name}"
        ica, t = run_ica(
            raw_input, method, params, reject, label=label, display_name=display_name
        )
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


# Build noisy datasets for each SNR level and noise type
noisy_sets = {}
idx = 0
for snr_db in snr_levels:
    for ntype in noise_types:
        raw_noisy_level, amp_ratio = add_noise_for_snr(
            raw_clean, snr_db, random_state=idx, noise_type=ntype
        )
        idx += 1
        # Scale reject thresholds based on noise amplitude ratio
        reject_scaled = dict(
            mag=reject_clean["mag"] * (1.0 + amp_ratio),
            grad=reject_clean["grad"] * (1.0 + amp_ratio),
        )
        label = f"noisy_{ntype}_snr{snr_db}dB"
        noisy_sets[label] = (raw_noisy_level, reject_scaled)

# Run on clean
icas_clean, times_clean, eog_clean = run_all_ica(raw_clean, "clean", reject_clean)

# Run on each noisy SNR level
icas_all = {**icas_clean}
times_all = {**times_clean}
eog_all = {**eog_clean}
for label, (raw_noisy_level, reject_scaled) in noisy_sets.items():
    icas_level, times_level, eog_level = run_all_ica(
        raw_noisy_level, label, reject_scaled
    )
    icas_all.update(icas_level)
    times_all.update(times_level)
    eog_all.update(eog_level)

# %%

# Clean EOG components for each algorithm (Column 1)
for method in ["fastica", "picard", "infomax", "infomax_extended"]:
    key = f"clean_{method}"
    comp = eog_all.get(key)
    if comp is not None:
        icas_all[key].plot_components(
            picks=[comp], title=f"{key} - EOG Component (Clean Data)", show=True
        )
        plt.close()

# %%

# Noisy EOG components for each algorithm at each SNR level and noise type
for label in noisy_sets.keys():
    for method in ["fastica", "picard", "infomax", "infomax_extended"]:
        key = f"{label}_{method}"
        comp = eog_all.get(key)
        if comp is not None:
            icas_all[key].plot_components(
                picks=[comp],
                title=f"{key} - EOG Component ({label.replace('_', ' ')})",
                show=True,
            )
            plt.close()
