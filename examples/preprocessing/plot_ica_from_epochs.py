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

###############################################################################
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
ecg_ch_name = 'MEG 1531'
ecg_scores = ica.find_sources_epochs(epochs, target=ecg_ch_name,
                                     score_func='pearsonr')

# get the source most correlated with the ECG.
ecg_source_idx = np.argsort(np.abs(ecg_scores))[-1]

# get sources as epochs object and inspect some trial
some_trial = 10
sources = ica.sources_as_epochs(epochs)
ecg_artifact = sources.get_data()[some_trial, ecg_source_idx]

# plot first epoch in ICA space
pl.figure()
pl.title('Source most correlated with the ECG channel')
pl.plot(sources.times, ecg_artifact, color='r')
pl.xlabel('Time (s)')
pl.ylabel('AU')
pl.show()

# As we have an EOG channel, we can use it to detect the source.
eog_scores = ica.find_sources_epochs(epochs, target='EOG 061',
                                     score_func='pearsonr')

# get maximum correlation index for EOG
eog_source_idx = np.abs(eog_scores).argmax()

# get sources as timeseries of concatenated epochs
sources = ica.get_sources_epochs(epochs, concatenate=True)
# compute times for concatenated epochs
times = np.linspace(epochs.times[0], epochs.times[-1] * len(epochs),
                    len(sources.T))

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
mags = mne.fiff.pick_types(epochs.info, meg='mag', exclude=[])
fig, axes = pl.subplots(2, 2, sharex=True, sharey=True)
times = epochs.times * 1e3
scale = 1e15
titles = ['raw - ', 'cleaned - ']
ecg_ch = epochs.ch_names.index(ecg_ch_name)
for e, (ax1, ax2), title in zip([epochs, epochs_ica], axes.T, titles):
    ax1.plot(times, e.average(mags).data.T * scale, color='k')
    ax1.set_title(title + 'evoked')
    ax2.plot(times, e._data[some_trial, ecg_ch].T * scale, color='r')
    ax2.set_title(title + 'single trial')
    if title == 'raw':
        ax1.set_ylabel('data (fT)')
    else:
        ax2.set_xlabel('Time (ms)')

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
