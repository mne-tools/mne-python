# -*- coding: utf-8 -*-
"""
.. _ex-csd-matrix:

=============================================
Compute a cross-spectral density (CSD) matrix
=============================================

A cross-spectral density (CSD) matrix is similar to a covariance matrix, but in
the time-frequency domain. It is the first step towards computing
sensor-to-sensor coherence or a DICS beamformer.

This script demonstrates the three methods that MNE-Python provides to compute
the CSD:

1. Using short-term Fourier transform: :func:`mne.time_frequency.csd_fourier`
2. Using a multitaper approach: :func:`mne.time_frequency.csd_multitaper`
3. Using Morlet wavelets: :func:`mne.time_frequency.csd_morlet`
"""
# Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
# License: BSD-3-Clause

# %%
import mne
from mne.datasets import sample
from mne.time_frequency import csd_fourier, csd_multitaper, csd_morlet

print(__doc__)

# %%
# In the following example, the computation of the CSD matrices can be
# performed using multiple cores. Set ``n_jobs`` to a value >1 to select the
# number of cores to use.
n_jobs = 1

# %%
# Loading the sample dataset.
data_path = sample.data_path()
meg_path = data_path / 'MEG' / 'sample'
fname_raw = meg_path / 'sample_audvis_raw.fif'
fname_event = meg_path / 'sample_audvis_raw-eve.fif'
raw = mne.io.read_raw_fif(fname_raw)
events = mne.read_events(fname_event)

# %%
# By default, CSD matrices are computed using all MEG/EEG channels. When
# interpreting a CSD matrix with mixed sensor types, be aware that the
# measurement units, and thus the scalings, differ across sensors. In this
# example, for speed and clarity, we select a single channel type:
# gradiometers.
picks = mne.pick_types(raw.info, meg='grad')

# Make some epochs, based on events with trigger code 1
epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=1,
                    picks=picks, baseline=(None, 0),
                    reject=dict(grad=4000e-13), preload=True)

# %%
# Computing CSD matrices using short-term Fourier transform and (adaptive)
# multitapers is straightforward:
csd_fft = csd_fourier(epochs, fmin=15, fmax=20, n_jobs=n_jobs)
csd_mt = csd_multitaper(epochs, fmin=15, fmax=20, adaptive=True, n_jobs=n_jobs)

# %%
# When computing the CSD with Morlet wavelets, you specify the exact
# frequencies at which to compute it. For each frequency, a corresponding
# wavelet will be constructed and convolved with the signal, resulting in a
# time-frequency decomposition.
#
# The CSD is constructed by computing the correlation between the
# time-frequency representations between all sensor-to-sensor pairs. The
# time-frequency decomposition originally has the same sampling rate as the
# signal, in our case ~600Hz. This means the decomposition is over-specified in
# time and we may not need to use all samples during our CSD computation, just
# enough to get a reliable correlation statistic. By specifying ``decim=10``,
# we use every 10th sample, which will greatly speed up the computation and
# will have a minimal effect on the CSD.
frequencies = [16, 17, 18, 19, 20]
csd_wav = csd_morlet(epochs, frequencies, decim=10, n_jobs=n_jobs)

# %%
# The resulting :class:`mne.time_frequency.CrossSpectralDensity` objects have a
# plotting function we can use to compare the results of the different methods.
# We're plotting the mean CSD across frequencies.
# :func:`mne.time_frequency.CrossSpectralDensity.plot()` returns a list of
# created figures; in this case, each returned list has only one figure
# so we use a Python trick of including a comma after our variable name
# to assign the figure (not the list) to our ``fig`` variable:
plot_dict = {'Short-time Fourier transform': csd_fft,
             'Adaptive multitapers': csd_mt,
             'Morlet wavelet transform': csd_wav}
for title, csd in plot_dict.items():
    fig, = csd.mean().plot()
    fig.suptitle(title)
