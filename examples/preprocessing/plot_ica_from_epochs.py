"""
================================
Compute ICA components on Epochs
================================

25 ICA components are estimated and displayed.

"""
print __doc__

# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import matplotlib.pylab as pl
import numpy as np
import mne
from mne.fiff import Raw
from mne.artifacts.ica import ICA
from mne.datasets import sample

data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = Raw(raw_fname, preload=True)

picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, eog=True,
                            ecg=True, stim=False, exclude=raw.info['bads'])

# setup ica seed
ica = ICA(noise_cov=None, n_components=25, random_state=0)
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
# remain hidden when plotting single epochs (#4 looks unsuspicious in the
# panel plot). However, plotting the identified source across epochs reveals
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

# join the detected artifact indices
exclude = np.r_[ecg_source_idx, eog_source_idx]

epochs_ica = ica.pick_sources_epochs(epochs, include=None, exclude=exclude,
                                     copy=True)

# plot original epochs
pl.figure()
epochs.average().plot()
pl.show()

# plot cleaned epochs
pl.figure()
epochs_ica.average().plot()
pl.show()
