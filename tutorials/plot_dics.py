# -*- coding: utf-8 -*-
"""
DICS for power mapping
======================

In this tutorial, we're going to simulate two signals originating from two
locations on the cortex. These signals will be sine waves, so we'll be looking
at oscillatory activity (as opposed to evoked activity).

We'll be using dynamic imaging of coherent sources (DICS) [1]_ to map out
spectral power along the cortex. Let's see if we can find our two simulated
sources.
"""
# Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
#
# License: BSD (3-clause)

###############################################################################
# Setup
# -----
# We first import the required packages to run this tutorial and define a list
# of filenames for various things we'll be using.
import os.path as op
import numpy as np
from scipy.signal import welch, coherence
from mayavi import mlab
from matplotlib import pyplot as plt

import mne
from mne.simulation import simulate_raw
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.time_frequency import csd_epochs
from mne.beamformer import make_dics, apply_dics_csd

# We use the MEG and MRI setup from the MNE-sample dataset
data_path = sample.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
mri_path = op.join(subjects_dir, 'sample')

# Filenames for various files we'll be using
meg_path = op.join(data_path, 'MEG', 'sample')
raw_fname = op.join(meg_path, 'sample_audvis_raw.fif')
trans_fname = op.join(meg_path, 'sample_audvis_raw-trans.fif')
src_fname = op.join(mri_path, 'bem', 'sample-oct-6-src.fif')
bem_fname = op.join(mri_path, 'bem', 'sample-5120-bem-sol.fif')
fwd_fname = op.join(meg_path, 'sample_audvis-meg-oct-6-fwd.fif')
cov_fname = op.join(meg_path, 'sample_audvis-cov.fif')

# Seed for the random number generator
rand = np.random.RandomState(42)

###############################################################################
# Data simulation
# ---------------
#
# The following function generates a timeseries that contains an oscillator,
# whose frequency fluctuates a little over time, but stays close to 10 Hz.
# We'll use this function to generate our two signals. Each signal will be
# zero for negative time, and have an oscillator in positive time.

sfreq = 100.  # Sampling frequency of the generated signal
base_freq = 10.  # Base frequency of the oscillators in Hertz
n_trials = 5
tmin = -0.5  # amount of zero signal to produce (sec)
tmax = 2.  # amount of oscillating signal to produce (sec)
times = np.arange(int(round(tmax * sfreq)) + 1) / sfreq


def coh_signal_gen():
    """Generate an oscillating signal.

    Returns
    -------
    signal : ndarray
        The generated signal.
    """
    t_rand = 1e-1 / sfreq  # Variation in the instantaneous frequency / phase
    std = 0.1  # Std-dev of the random fluctuations added to the signal
    n_times = len(times)

    # Generate an oscillator with varying frequency and phase lag.
    signal = np.sin(2.0 * np.pi *
                    (base_freq * np.arange(n_times) / sfreq +
                     np.cumsum(t_rand * rand.randn(n_times))))

    # Add some random fluctuations to the signal.
    signal += std * rand.randn(n_times)

    # Scale the signal to be in the right order of magnitude (~250 nAm)
    # to achieve a SNR of 1 with our noise covariance matrix.
    signal *= 250e-9

    return signal


###############################################################################
# Let's simulate two timeseries and plot some basic information about them.
signal1 = coh_signal_gen()
signal2 = coh_signal_gen()

fig, axes = plt.subplots(2, 2, figsize=(8, 4))

# Plot the timeseries
ax = axes[0][0]
ax.plot(times, 1e9 * signal1, lw=0.5)
ax.set(xlabel='Time (s)', xlim=times[[0, -1]], ylabel='Amplitude (Am)',
       title='Signal 1')
ax = axes[0][1]
ax.plot(times, 1e9 * signal2, lw=0.5)
ax.set(xlabel='Time (s)', xlim=times[[0, -1]], title='Signal 2')

# Power spectrum of the first timeseries
freq_kwargs = dict(fs=sfreq, nperseg=50, noverlap=25, detrend=False)
f, p = welch(signal1, **freq_kwargs)
ax = axes[1][0]
ax.plot(f, 20 * np.log10(p), lw=1.)
ax.set(xlabel='Frequency (Hz)', xlim=f[[0, -1]],
       ylabel='Power (dB)', title='Power spectrum of signal 1')

# Compute the coherence between the two timeseries
f, coh = coherence(signal1, signal2, **freq_kwargs)
ax = axes[1][1]
ax.plot(f, coh, lw=1.)
ax.set(xlabel='Frequency (Hz)', xlim=f[[0, -1]], ylabel='Coherence',
       title='Coherence between the timeseries')
fig.tight_layout()

###############################################################################
# Now we put the signals at two locations on the cortex. We construct a
# :class:`mne.SourceEstimate` object to store them in.
#
# The timeseries will have a part where the signal is active and a part where
# it is not. The techniques we'll be using in this tutorial depend on being
# able to contrast data that contains the signal of interest versus data that
# does not (i.e. it contains only noise).

# The locations on the cortex where the signal will originate from. These
# locations are indicated as vertex numbers.
source_vert1 = 146374
source_vert2 = 33830

# The timeseries at each vertex: one part zeros, one part signal
n_noise = int(round(-tmin * sfreq))
timeseries1 = np.hstack([np.zeros(n_noise), signal1])
timeseries2 = np.hstack([np.zeros(n_noise), signal2])

# Construct a SourceEstimate object that describes the signal at the cortical
# level.
tstep = 1. / sfreq
stc = mne.SourceEstimate(
    np.vstack((timeseries1, timeseries2)),  # The two timeseries
    vertices=[[source_vert1], [source_vert2]],  # Their locations
    tmin=tmin, tstep=tstep, subject='sample',
)

###############################################################################
# Before we simulate the sensor-level data, let's define a signal-to-noise
# ratio. You are encouraged to play with this parameter and see the effect of
# noise on our results.
snr = 1.  # Signal-to-noise ratio. Decrease to add more noise.

###############################################################################
# Now we run the signal through the forward model to obtain simulated sensor
# data. To save computation time, we'll only simulate gradiometer data. You can
# try simulating other types of sensors as well.
#
# Some noise is added based on the baseline noise covariance matrix from the
# sample dataset, scaled to implement the desired SNR.

cov = mne.read_cov(cov_fname)
cov['data'] /= snr ** 2

# This is the raw file we're going to use as template for the simulated data.
# Here we will restrict to gradiometers for speed.
info = mne.io.read_info(raw_fname)
info.update(sfreq=sfreq, bads=[])
picks = mne.pick_types(info, meg='grad', stim=True, exclude=())
mne.pick_info(info, picks, copy=False)
raw = mne.io.RawArray(np.zeros((len(info['ch_names']),
                                stc.data.shape[1] * n_trials)), info)

# Simulate the raw data, with a lowpass filter on the noise
raw = simulate_raw(raw, stc, trans_fname, src_fname, bem_fname, cov=cov,
                   random_state=rand, iir_filter=[4, -4, 0.8])

###############################################################################
# We create an :class:`mne.Epochs` object containing trials where positive
# time contains both noise and signal, and negative time only noise.

events = mne.find_events(raw, 'STI 014')
assert len(events) == n_trials
epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, preload=True)

# Plot the simulated data
epochs.plot()

# crop to just our region with signal
epochs.crop(0., None)

###############################################################################
# Power mapping
# -------------
# With our simulated dataset ready, we can now pretend to be researchers that
# have just recorded this from a real subject and are going to study what parts
# of the brain communicate with each other.
#
# First, we'll create a source estimate of the MEG data. We'll use both a
# straightforward MNE-dSPM inverse solution for this, and the DICS beamformer
# which is specifically designed to work with oscillatory data.

###############################################################################
# Computing the inverse using MNE-dSPM:

# Estimating the noise covariance on the trial that only contains noise.
fwd = mne.read_forward_solution(fwd_fname)
inv = make_inverse_operator(info, fwd, cov)

# Apply the inverse model to the trial that also contains the signal.
s = apply_inverse(epochs.average(), inv)

# Take the root-mean square along the time dimension and plot the result.
s_rms = np.sqrt((s ** 2).mean())
brain = s_rms.plot('sample', subjects_dir=subjects_dir, hemi='both', figure=1,
                   size=600)

# Indicate the true locations of the source activity on the plot.
brain.add_foci(source_vert1, coords_as_verts=True, hemi='lh')
brain.add_foci(source_vert2, coords_as_verts=True, hemi='rh')

# Rotate the view and add a title.
mlab.view(0, 0, 550, [0, 0, 0])
title = mlab.title('dSPM inverse\n(RMS)', height=0.9)

###############################################################################
# Computing a cortical power map at the base frequency using a DICS beamformer:

# Estimate the cross-spectral density (CSD) matrix on the trial containing the
# signal.
csd_signal = csd_epochs(epochs, mode='cwt_morlet',
                        frequencies=[base_freq])

# Compute the DICS powermap. An important parameter for this is the
# regularization, which is set quite high for this toy example. For real data,
# you may want to lower this to around 0.05.
filters = make_dics(info, fwd, csd_signal, reg=0.05, pick_ori='max-power')
power, f = apply_dics_csd(csd_signal, filters)

# Plot the DICS power map.
brain = power.plot('sample', subjects_dir=subjects_dir, hemi='both', figure=2,
                   size=600)

# Indicate the true locations of the source activity on the plot.
brain.add_foci(source_vert1, coords_as_verts=True, hemi='lh')
brain.add_foci(source_vert2, coords_as_verts=True, hemi='rh')

# Rotate the view and add a title.
mlab.view(0, 0, 550, [0, 0, 0])
title = mlab.title('DICS power map\n%.1f Hz' % f[0], height=0.9)

###############################################################################
# Excellent! Both methods found our two simulated sources. Of course, with a
# signal-to-noise ratio (SNR) of 1, is isn't very hard to find them. You can
# try playing with the SNR and see how the MNE-dSPM and DICS results hold up in
# the presence of increasing noise.

###############################################################################
# References
# ----------
# .. [1] Gross, J., Kujala, J., Hamalainen, M., Timmermann, L., Schnitzler, A.,
#    & Salmelin, R. (2001). Dynamic imaging of coherent sources: Studying
#    neural interactions in the human brain. Proceedings of the National
#    Academy of Sciences, 98(2), 694–699. https://doi.org/10.1073/pnas.98.2.694
