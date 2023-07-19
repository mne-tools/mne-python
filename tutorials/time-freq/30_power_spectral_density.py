"""
.. _tut-sensors-psd:

====================================================
Power Spectral Density Analysis: Spectral Decoupling
====================================================

The objective of this tutorial is describe the basics of power spectral
density and what it can tell us about underlying brain activity. Power spectral
density represents time-series data as the magnitude of sine and cosine
coefficients of the Fourtier Transform; how much of each different
frequency of sinusoidal wave best represents your time-series data. While,
understandably, this method is great at detecting oscillatory neural activity
(activity that waxes and wanes periodically at a particular frequency
or rhythm), interestingly, it also yields important information about
aperiodic neural activity through the background or broadband changes in power.
Brain activity is consistently observed to have exponentially decreasing
background power, like pink noise, with oscillatory peaks superimposed,
like the peaks in a nuclear magnetic resonance (NMR) spectroscopy scan. The
peaks can tell us about oscillatory (periodic), synchronous brain activity
and the background power can tell us about non-oscillatory (aperiodic),
asynchronous brain activity. (For contrast, an event-related potential,
the deflection in an electrophysiology recording after an event is shown,
is aperiodic because it doesn't repeat but synchronous because it is
synchronized by the event). This tutorial will demonstrate how this
interpretation of power spectral density can be used to study a movement
task. Unfortunately, since this method was demonstrated on
electrocortigraphy (ECoG) which can discriminate the location of brain
activity at much better resolution than scalp electroencephalography (EEG),
we won't be able to show that, during movement, the broadband changes
in power are confined to a more specific brain area (the brain area
that controls that particular movement) whereas the oscillation is
spread across a large portion of primary motor cortex (the gyrus that,
when stimulated, causes movement of different body parts depending
on the location of stimulation) but please read :footcite:`MillerEtAl2009A`
for more details.
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
raw.set_montage("standard_1005")

# make epochs
events, event_id = mne.events_from_annotations(raw)
# four seconds of movement/rest; start 1 s after onset and end 1s before
# offset for just movement/rest
epochs = mne.Epochs(
    raw, events, tmin=1, tmax=3, reject=dict(eeg=4e-4), baseline=None, preload=True
)

# %%
# First, let's compute the power spectral density and plot it.
psd = epochs.compute_psd(fmax=75)
psd.plot()

# %%
# There is a very large artifact in our signal from the power supply
# to the building where the data was collected, let's remove that with
# a notch filter so that it won't dominate our signal.
raw.notch_filter([60])
epochs = mne.Epochs(
    raw, events, tmin=1, tmax=3, reject=dict(eeg=4e-4), baseline=None, preload=True
)
psd = epochs.compute_psd(fmax=75)
psd.plot()

# %%
# Now, there are two main components to a power spectrum: 1) The
# power that is present across all frequencies and decreases
# exponentially at higher frequencies (called the 1/f component
# or power law scaling or broadband power) and 2) peaks, generally with a
# normal distribution above this background power. The broadband power
# reflects neural activity that is aperiod and asynchronous; when broadband
# power is greater, more neurons are firing total but that they are not
# synchronized with each other in an oscillatory rhythm :footcite:`ManningEtAl2009`.
# Peaks in the power spectrum, on the other hand, are interpreted
# as periodic, synchronous neural activity.

# %%
# We can separate out these using principal component analysis (PCA) as in
# :footcite:`MillerEtAl2009A`. Let's see how this works:

# select the only channel so the data is (epochs x freqs)
psd_data = psd.get_data(picks=["C3"])[:, 0]

# normalize
psd_data = np.log(psd_data) - np.log(psd_data.mean(axis=1, keepdims=True))

# prepare to remove frequencies contaminated by line noise
mask = np.logical_or(psd.freqs < 57, psd.freqs > 63)

# set a random seed for reproducibility
pca = PCA(svd_solver="randomized", whiten=True, random_state=99).fit(psd_data[:, mask])

# %%
# As shown below, the maroon component (1st principal component (PC)) has weights evenly
# spread across frequencies whereas the tan component (2nd PC) is peaked at around 16 Hz
# which is considered in the beta (13 - 30 Hz) band of frequencies.
# Admittedly, this is not as clean in scalp electroencephalography (EEG) as it is in
# electrocorticography (ECoG) as was done in the paper referenced. ECoG is implanted
# on the surface of the brain so it detects more brain signal.
fig, ax = plt.subplots()
comp0 = np.zeros((psd.freqs.size,)) * np.nan
comp0[mask] = pca.components_[0]
ax.plot(psd.freqs, comp0, color="maroon")
comp1 = np.zeros((psd.freqs.size,)) * np.nan
comp1[mask] = pca.components_[1]
ax.plot(psd.freqs, comp1, color="tan")
ax.axhline(0)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Component Weight")

# %%
# One thing to notice is that the principal components tend to be generally in opposite
# directions. This is likely because principal component are required to be orthogonal.
# This is not the case for factor analysis, which is like PCA without orthogonal axes.
# Notice that the first and second PCs mirror each other less across the ``y=0`` line.
psd_data = psd.get_data(picks=["C3"])[:, 0]
psd_data = np.log(psd_data) - np.log(psd_data.mean(axis=1, keepdims=True))
mask = np.logical_or(psd.freqs < 57, psd.freqs > 63)
fa = FactorAnalysis(rotation="varimax", random_state=99).fit(psd_data[:, mask])

fig, ax = plt.subplots()
comp0 = np.zeros((psd.freqs.size,)) * np.nan
comp0[mask] = fa.components_[0]
ax.plot(psd.freqs, comp0, color="maroon")
comp1 = np.zeros((psd.freqs.size,)) * np.nan
comp1[mask] = fa.components_[1]
ax.plot(psd.freqs, comp1, color="tan")
ax.axhline(0)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Component Weight")

# Finally, let's apply the PCA to our data and see if this helps us separate
# movement epochs from rest epochs.
event_mask = [entry == () for entry in epochs.drop_log]
move_events = np.logical_or(
    events[:, 2] == event_id["T1"], events[:, 2] == event_id["T2"]
)[event_mask]
rest_events = (events[:, 2] == event_id["T0"])[event_mask]

# %%
# We see that we are indeed able to recapitulate the figures from
# :footcite:`MillerEtAl2009A` with a bit weaker effects using scalp EEG than
# ECOG. Note particularly that, as you get into higher frequencies,
# the power spectra for the two conditions are parallel.
# Where there are more oscillations, in the lower frequencies
# (below 30 Hz), this becomes obscured, but, in :footcite:`MillerEtAl2009B` higher
# frequencies are expolored using ECoG and basically this phenoma holds out at
# those higher frequencies indicating that the connectivity of the brain probably
# doesn't change fundamentally but rather this broadband shape shifts up and
# down when more or fewer neurons are firing total near the recording site.
#
# Finally, also note that in :footcite:`MillerEtAl2009A`, the ECoG grid
# covered the regions of primary motor cortex responsible for multiple
# movements, whereas the C3 electrode is roughly over primary motor cortex
# and so records the activity of a relatively large area of primary motor
# cortex, spanning areas that control different limbs. Because of this, we
# are unable to see that the broadband shifts occur focally in the
# primary motor cortex region that controls the particular movement whereas
# the beta desynchronization is more widespread across most of primary
# motor cortex. This is evidence that ties into the spotlight hypothesis
# of motor control where widespread inhibition of the motor system (which seems
# to be mediated by this beta oscillation decrease) facilitates choosing the
# correct response (potentially mediated by the broadband power increase)
# like quieting a crowd in a stadium order to pick one person out in
# particular :footcite:`GreenhouseEtAl2015`.
fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 10))
ax.set_title("Full Recording")
move_psd_data = np.zeros((psd.freqs.size,)) * np.nan
move_psd_data[mask] = psd_data[move_events].mean(axis=0)[mask]
ax.plot(psd.freqs, move_psd_data, color="green", linewidth=0.5)
rest_psd_data = np.zeros((psd.freqs.size,)) * np.nan
rest_psd_data[mask] = psd_data[rest_events].mean(axis=0)[mask]
ax.plot(psd.freqs, rest_psd_data, color="black", linewidth=0.5)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel(r"Power ($\mu$$V^2$)")

psd_mean = psd_data[:, mask].mean(axis=0)

ax2.set_title("1st PC (Broadband Power)")
move_psd_data = np.zeros((psd.freqs.size,)) * np.nan
move_psd_data[mask] = np.mean(
    np.dot(pca.transform(psd_data[move_events][:, mask])[:, 0:1], pca.components_[0:1])
    + psd_mean,
    axis=0,
)
ax2.plot(psd.freqs, move_psd_data, color="green", linewidth=0.5)
rest_psd_data = np.zeros((psd.freqs.size,)) * np.nan
rest_psd_data[mask] = np.mean(
    np.dot(pca.transform(psd_data[rest_events][:, mask])[:, 0:1], pca.components_[0:1])
    + psd_mean,
    axis=0,
)
ax2.plot(psd.freqs, rest_psd_data, color="black", linewidth=0.5)
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel(r"Power ($\mu$$V^2$)")

ax3.set_title("2nd PC (Beta Oscillations)")
move_psd_data = np.zeros((psd.freqs.size,)) * np.nan
move_psd_data[mask] = np.mean(
    np.dot(pca.transform(psd_data[move_events][:, mask])[:, 1:2], pca.components_[1:2])
    + psd_mean,
    axis=0,
)
ax3.plot(psd.freqs, move_psd_data, color="green", linewidth=0.5)
rest_psd_data = np.zeros((psd.freqs.size,)) * np.nan
rest_psd_data[mask] = np.mean(
    np.dot(pca.transform(psd_data[rest_events][:, mask])[:, 1:2], pca.components_[1:2])
    + psd_mean,
    axis=0,
)
ax3.plot(psd.freqs, rest_psd_data, color="black", linewidth=0.5)
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel(r"Power ($\mu$$V^2$)")
fig.tight_layout()
