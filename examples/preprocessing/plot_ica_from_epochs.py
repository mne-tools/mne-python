"""
================================
Compute ICA components on Epochs
================================

ICA is used to decompose raw data in 49 to 50 sources.
The source matching the ECG is found automatically
and displayed. Finally, the cleaned epochs are compared
to the uncleaned epochs.

"""
print __doc__

# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import matplotlib.pylab as pl
import numpy as np
import mne
from mne.fiff import Raw
from mne.preprocessing.ica import ICA
from mne.datasets import sample

###############################################################################
# Setup paths and prepare epochs data

data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = Raw(raw_fname, preload=True)

picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, eog=True,
                            ecg=True, stim=False, exclude=raw.info['bads'])

###############################################################################
# Setup ICA seed decompose data, then access and plot sources.

# Instead of the actual number of components here we pass a float value
# between 0 and 1 to select n_components by a percentage of
# explained variance. Also we decide to use 64 PCA components before mixing
# back to sensor space. These include the PCA components supplied to ICA plus
# additional PCA components up to rank 64 of the MEG data.
# This allows to control the trade-off between denoising and preserving signal.

ica = ICA(n_components=0.90, n_pca_components=64, max_pca_components=100,
          noise_cov=None, random_state=0)
print ica

# get epochs
tmin, tmax, event_id = -0.2, 0.5, 1
# baseline = None
baseline = (None, 0)
reject = None

events = mne.find_events(raw, stim_channel='STI 014')
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=baseline, preload=True, reject=reject)


# fit sources from epochs or from raw (both works for epochs)
ica.decompose_epochs(epochs)

# plot components for one epoch of interest
# A distinct cardiac component should be visible
ica.plot_sources_epochs(epochs, epoch_idx=13, n_components=25)

###############################################################################
# Automatically find the ECG component using correlation with ECG signal

# As we don't have an ECG channel we use one that correlates a lot with heart
# beats: 'MEG 1531'. We can directly pass the name to the find_sources method.
# In our example, the find_sources method returns and array of correlation
# scores for each ICA source.

ecg_scores = ica.find_sources_epochs(epochs, target='MEG 1531',
                                     score_func='pearsonr')

# get maximum correlation index for ECG
ecg_source_idx = np.abs(ecg_scores).argmax()

print '#%i -- ICA component resembling the ECG' % ecg_source_idx

###############################################################################
# Automatically find the EOG component using correlation with EOG signal

# As we have an EOG channel, we can use it to detect the source.

eog_scores = ica.find_sources_epochs(epochs, target='EOG 061',
                                     score_func='pearsonr')

# get maximum correlation index for EOG
eog_source_idx = np.abs(eog_scores).argmax()

print '#%i -- ICA component resembling the EOG' % eog_source_idx

# As the subject did not constantly move her eyes, the movement artifacts
# may remain hidden when plotting single epochs.
# Plotting the identified source across epochs reveals
# considerable EOG artifacts.

# get maximum correlation index for EOG
eog_source_idx = np.abs(eog_scores).argmax()

# get sources
sources = ica.get_sources_epochs(epochs, concatenate=True)

pl.figure()
pl.title('Source most correlated with the EOG channel')
pl.plot(sources[eog_source_idx].T)
pl.show()

###############################################################################
# Reject artifact sources and compare results

# Add the detected artifact indices to ica.exclude
ica.exclude += [ecg_source_idx, eog_source_idx]

# Restore sensor space data
epochs_ica = ica.pick_sources_epochs(epochs, include=None)

# plot original epochs
pl.figure()
epochs.average().plot()
pl.show()

# plot cleaned epochs
pl.figure()
epochs_ica.average().plot()
pl.show()
