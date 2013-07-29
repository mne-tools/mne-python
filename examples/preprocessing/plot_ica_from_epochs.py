"""
================================
Compute ICA components on epochs
================================

ICA is used to decompose raw data in 49 to 50 sources.
The source matching the ECG is found automatically
and displayed. Finally, after the cleaned epochs are
compared to the uncleaned epochs, evoked ICA sources
are investigated using sensor space ERF plotting
techniques.

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

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = Raw(raw_fname, preload=True)
raw.apply_proj()

picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, eog=True,
                            ecg=True, stim=False, exclude='bads')

tmin, tmax, event_id = -0.2, 0.5, 1
baseline = (None, 0)
reject = None

events = mne.find_events(raw, stim_channel='STI 014')
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False, picks=picks,
                    baseline=baseline, preload=True, reject=reject)

random_state = np.random.RandomState(42)

#####################################################################################
# Setup ICA seed decompose data, then access and plot sources.
# for more background information visit the plot_ica_from_raw.py example

# fit sources from epochs or from raw (both works for epochs)
ica = ICA(n_components=0.90, n_pca_components=64, max_pca_components=100,
          noise_cov=None, random_state=random_state)

ica.decompose_epochs(epochs)
print ica

# plot spatial sensitivities of a few ICA components
title = 'Spatial patterns of ICA components (Magnetometers)'
source_idx = range(35, 50)
ica.plot_topomap(source_idx, ch_type='mag')
pl.suptitle(title, fontsize=12)


###############################################################################
# Automatically find ECG and EOG component using correlation coefficient.

# As we don't have an ECG channel we use one that correlates a lot with heart
# beats: 'MEG 1531'. We can directly pass the name to the find_sources method.
# In our example, the find_sources method returns and array of correlation
# scores for each ICA source.

ecg_scores = ica.find_sources_epochs(epochs, target='MEG 1531',
                                     score_func='pearsonr')

# get maximum correlation index for ECG
ecg_source_idx = np.abs(ecg_scores).argmax()

# get sources from concatenated epochs
sources = ica.get_sources_epochs(epochs, concatenate=True)

# plot first epoch
times = epochs.times
first_trial = np.arange(len(times))

pl.figure()
pl.title('Source most correlated with the ECG channel')
pl.plot(times, sources[ecg_source_idx, first_trial].T, color='r')
pl.xlabel('Time (s)')
pl.ylabel('AU')
pl.show()

# As we have an EOG channel, we can use it to detect the source.
eog_scores = ica.find_sources_epochs(epochs, target='EOG 061',
                                     score_func='pearsonr')

# get maximum correlation index for EOG
eog_source_idx = np.abs(eog_scores).argmax()

# compute times for concatenated epochs
times = np.linspace(times[0], times[-1] * len(epochs), sources.shape[1])

# As the subject did not constantly move her eyes, the movement artifacts
# may remain hidden when plotting single epochs.
# Plotting the identified source across epochs reveals
# considerable EOG artifacts.

pl.figure()
pl.title('Source most correlated with the EOG channel')
pl.plot(times, sources[eog_source_idx].T, color='r')
pl.xlabel('Time (s)')
pl.ylabel('AU')
pl.xlim(times[[0, -1]])
pl.show()


###############################################################################
# Reject artifact sources and compare results

# Add detected artifact sources to exclusion list
ica.exclude += [ecg_source_idx, eog_source_idx]

# Restore sensor space data
epochs_ica = ica.pick_sources_epochs(epochs)


# First show unprocessed, then cleaned epochs
for e in epochs, epochs_ica:
    pl.figure()
    e.average().plot()
    pl.show()

###############################################################################
# Inspect evoked ICA sources

# create ICA Epochs object.
ica_epochs = ica.sources_as_epochs(epochs)

# don't exclude bad sources by passing an empty list.
ica_picks = mne.fiff.pick_types(ica_epochs.info, misc=True, exclude=[])
ica_evoked = ica_epochs.average(ica_picks)
pl.figure()
ica_evoked.plot(titles=dict(misc='ICA sources'))

# Tip: use this for epochs constructed around ECG r-peaks to check whether all
# ECG components were identified.
