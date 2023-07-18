"""
.. _tut-sensors-psd:

===============================
Power Spectral Density Analysis
===============================

The objective of this tutorial is describe the basics of power spectral
density and what it can tell us about underlying brain activity.
"""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis

import mne
from mne.datasets import eegbci

# %%
# Load the data: from :ref:`ex-eeg-bridging` we can see that subject 5
# has no bridging and from the documentation in :func:`mne.datasets.eegbci.load_data`
# we can pick the runs where the subject was performing a movement task
# (3, 5, 7, 9, 11, 13).
raw_fnames = eegbci.load_data(subject=5, runs=[3, 5, 7, 9, 11, 13])
raws = [mne.io.read_raw(f, preload=True) for f in raw_fnames]

# join files and standardize format
raw = mne.concatenate_raws(raws)
eegbci.standardize(raw)  # set channel names
montage = mne.channels.make_standard_montage("standard_1005")
raw.set_montage(montage)

# make epochs
events, event_id = mne.events_from_annotations(raw)
# four seconds of movement/rest; start 1 s after onset and end 1s before
# offset for just movement/rest
epochs = mne.Epochs(raw, events, tmin=1, tmax=3,
                    reject=dict(eeg=4e-4), baseline=None, preload=True)

# %%
# First, let's compute the power spectral density and plot it.
psd = epochs.compute_psd(fmax=75)
psd.plot()

# %%
# There is a very large artifact in our signal from the power supply
# to the building where the data was collected, let's remove that with
# a notch filter so that it won't dominate our signal.
raw.notch_filter([60])
epochs = mne.Epochs(raw, events, tmin=1, tmax=3,
                    reject=dict(eeg=4e-4), baseline=None, preload=True)
psd = epochs.compute_psd(fmax=75)
psd.plot()

# %%
# Now, there are two main components to a power spectrum: 1) The
# power that is present across all frequencies and decreases
# exponentially at higher frequencies and 2) Peaks, generally with a
# normal distribution above this background power. We can separate
# out these using principle component analysis (PCA) as in
# :footcite:`MillerEtAl2009A`. Let's see how this works:

# select the only channel so the data is (epochs x freqs)
psd_data = psd.get_data(picks=['C3'])[:, 0]

# normalize
psd_data = np.log(psd_data) - np.log(psd_data.mean(axis=1, keepdims=True))

# prepare to remove frequencies contaminated by line noise
mask = np.logical_or(psd.freqs < 57, psd.freqs > 63)

# set a random seed for reproducibility
pca = PCA(svd_solver='randomized', whiten=True, random_state=99).fit(psd_data[:, mask])

fig, ax = plt.subplots()
comp0 = np.zeros((psd.freqs.size,)) * np.nan
comp0[mask] = pca.components_[0]
ax.plot(psd.freqs, comp0, color='maroon')
comp1 = np.zeros((psd.freqs.size,)) * np.nan
comp1[mask] = pca.components_[1]
ax.plot(psd.freqs, comp1, color='tan')
fig.show()

psd_data = psd.get_data(picks=['C3'])[:, 0]
psd_data = np.log(psd_data) - np.log(psd_data.mean(axis=1, keepdims=True))
n_epochs, n_freqs = psd_data.shape

# make into EpochsArray with frequencies instead of times
raw_psd = mne.io.RawArray(psd_data, mne.create_info(
    [f'ch{i}' for i in range(n_epochs)], psd.info['sfreq'], 'eeg'))

ica = mne.preprocessing.ICA(n_components=5).fit(raw_psd)
sources_psd = ica.get_sources(raw_psd).get_data()

fig, ax = plt.subplots()
ax.plot(psd.freqs, sources_psd[0], color='maroon')
ax.plot(psd.freqs, sources_psd[1], color='tan')
fig.show()
