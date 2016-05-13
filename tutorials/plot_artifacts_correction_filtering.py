"""
.. _tut_artifacts_filter:

Filtering and Resampling
========================

Certain artifacts are restricted to certain frequencies and can therefore
be fixed by filtering. An artifact that typically affects only some
frequencies is due to the power line.

Power-line noise is a noise created by the electrical network.
It is composed of sharp peaks at 50Hz (or 60Hz depending on your
geographical location). Some peaks may also be present at the harmonic
frequencies, i.e. the integer multiples of
the power-line frequency, e.g. 100Hz, 150Hz, ... (or 120Hz, 180Hz, ...).
"""

import numpy as np
import mne
from mne.datasets import sample

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
proj_fname = data_path + '/MEG/sample/sample_audvis_eog_proj.fif'

tmin, tmax = 0, 20  # use the first 20s of data

# Setup for reading the raw data (save memory by cropping the raw data
# before loading it)
raw = mne.io.read_raw_fif(raw_fname).crop(tmin, tmax).load_data()
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # bads + 2 more

fmin, fmax = 2, 300  # look at frequencies between 2 and 300Hz
n_fft = 2048  # the FFT size (n_fft). Ideally a power of 2

# Pick a subset of channels (here for speed reason)
selection = mne.read_selection('Left-temporal')
picks = mne.pick_types(raw.info, meg='mag', eeg=False, eog=False,
                       stim=False, exclude='bads', selection=selection)

# Let's first check out all channel types
raw.plot_psd(area_mode='range', tmax=10.0, picks=picks)

###############################################################################
# Removing power-line noise with notch filtering
# ----------------------------------------------
#
# Removing power-line noise can be done with a Notch filter, directly on the
# Raw object, specifying an array of frequency to be cut off:

raw.notch_filter(np.arange(60, 241, 60), picks=picks)
raw.plot_psd(area_mode='range', tmax=10.0, picks=picks)

###############################################################################
# Removing power-line noise with low-pas filtering
# -------------------------------------------------
#
# If you're only interested in low frequencies, below the peaks of power-line
# noise you can simply low pass filter the data.

raw.filter(None, 50.)  # low pass filtering below 50 Hz
raw.plot_psd(area_mode='range', tmax=10.0, picks=picks)

###############################################################################
# High-pass filtering to remove slow drifts
# -----------------------------------------
#
# If you're only interested in low frequencies, below the peaks of power-line
# noise you can simply high pass filter the data.

raw.filter(1., None)  # low pass filtering above 1 Hz
raw.plot_psd(area_mode='range', tmax=10.0, picks=picks)

###############################################################################
# To do the low-pass and high-pass filtering in one step you can do
# a so-called *band-pass* filter by running

raw.filter(1., 50.)  # band-pass filtering in the range 1 Hz - 50 Hz

###############################################################################
# Down-sampling (for performance reasons)
# ---------------------------------------
#
# When performing experiments where timing is critical, a signal with a high
# sampling rate is desired. However, having a signal with a much higher
# sampling rate than necessary needlessly consumes memory and slows down
# computations operating on the data. To avoid that, you can down-sample
# your time series.
#
# Data resampling can be done with *resample* methods.

raw.resample(100, npad="auto")  # set sampling frequency to 100Hz
raw.plot_psd(area_mode='range', tmax=10.0, picks=picks)

###############################################################################
# Since down-sampling reduces the timing precision of events, you might want to
# first extract epochs and down-sampling the Epochs object. You can do this
# using the :func:`mne.Epochs.resample` method.
