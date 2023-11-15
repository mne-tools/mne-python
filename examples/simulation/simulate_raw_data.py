"""
.. _ex-sim-raw:

===========================
Generate simulated raw data
===========================

This example generates raw data by repeating a desired source activation
multiple times.
"""
# Authors: Yousra Bekhti <yousra.bekhti@gmail.com>
#          Mark Wronkiewicz <wronk.mark@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

# %%

import matplotlib.pyplot as plt
import numpy as np

import mne
from mne import Epochs, compute_covariance, find_events, make_ad_hoc_cov
from mne.datasets import sample
from mne.simulation import (
    add_ecg,
    add_eog,
    add_noise,
    simulate_raw,
    simulate_sparse_stc,
)

print(__doc__)

data_path = sample.data_path()
meg_path = data_path / "MEG" / "sample"
raw_fname = meg_path / "sample_audvis_raw.fif"
fwd_fname = meg_path / "sample_audvis-meg-eeg-oct-6-fwd.fif"

# Load real data as the template
raw = mne.io.read_raw_fif(raw_fname)
raw.set_eeg_reference(projection=True)

##############################################################################
# Generate dipole time series
n_dipoles = 4  # number of dipoles to create
epoch_duration = 2.0  # duration of each epoch/event
n = 0  # harmonic number
rng = np.random.RandomState(0)  # random state (make reproducible)


def data_fun(times):
    """Generate time-staggered sinusoids at harmonics of 10Hz."""
    global n
    n_samp = len(times)
    window = np.zeros(n_samp)
    start, stop = [
        int(ii * float(n_samp) / (2 * n_dipoles)) for ii in (2 * n, 2 * n + 1)
    ]
    window[start:stop] = 1.0
    n += 1
    data = 25e-9 * np.sin(2.0 * np.pi * 10.0 * n * times)
    data *= window
    return data


times = raw.times[: int(raw.info["sfreq"] * epoch_duration)]
fwd = mne.read_forward_solution(fwd_fname)
src = fwd["src"]
stc = simulate_sparse_stc(
    src, n_dipoles=n_dipoles, times=times, data_fun=data_fun, random_state=rng
)
# look at our source data
fig, ax = plt.subplots(1)
ax.plot(times, 1e9 * stc.data.T)
ax.set(ylabel="Amplitude (nAm)", xlabel="Time (s)")
mne.viz.utils.plt_show()

##############################################################################
# Simulate raw data
raw_sim = simulate_raw(raw.info, [stc] * 10, forward=fwd, verbose=True)
cov = make_ad_hoc_cov(raw_sim.info)
add_noise(raw_sim, cov, iir_filter=[0.2, -0.2, 0.04], random_state=rng)
add_ecg(raw_sim, random_state=rng)
add_eog(raw_sim, random_state=rng)
raw_sim.plot()

##############################################################################
# Plot evoked data
events = find_events(raw_sim)  # only 1 pos, so event number == 1
epochs = Epochs(raw_sim, events, 1, tmin=-0.2, tmax=epoch_duration)
cov = compute_covariance(
    epochs, tmax=0.0, method="empirical", verbose="error"
)  # quick calc
evoked = epochs.average()
evoked.plot_white(cov, time_unit="s")
