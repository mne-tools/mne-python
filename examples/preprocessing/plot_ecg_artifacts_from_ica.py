"""
================================
Find ECG events from ICA sources
================================

ICA is used to decompose raw data in 49 to 50 sources.
The source matching the ECG is found automatically
identified and then used to detect ECG artifacts in
the raw data.

"""
print __doc__

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import numpy as np
import pylab as pl

import mne
from mne.fiff import Raw
from mne.preprocessing.ica import ICA, ica_find_ecg_events
from mne.datasets import sample

data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = Raw(raw_fname, preload=True)

picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, eog=False,
                            stim=False, exclude=raw.info['bads'])

###############################################################################
# Setup ICA seed decompose data, then access and plot sources.

# Sign and order of components is non deterministic.
# setting the random state to 0 makes the solution reproducible.
# Instead of the actual number of components we pass a float value
# between 0 and 1 to select n_components by a percentage of
# explained variance.

ica = ICA(n_components=0.90, max_pca_components=100, noise_cov=None,
          random_state=0)

# For maximum rejection performance we will compute the decomposition on
# the entire time range

# decompose sources for raw data, select n_components by explained variance
ica.decompose_raw(raw, start=None, stop=None, picks=picks)
print ica

# setup reasonable time window for inspection
start_plot, stop_plot = raw.time_as_index([100, 103])

# plot components
ica.plot_sources_raw(raw, start=start_plot, stop=stop_plot)

###############################################################################
# Automatically find the ECG component using correlation with ECG signal

# As we don't have an ECG channel we use one that correlates a lot with heart
# beats: 'MEG 1531'. We can directly pass the name to the find_sources method.
# We select the pearson correlation from scipy stats via string label.
# The function is internally modified to be applicable to 2D arrays and,
# hence, returns product-moment correlation scores for each ICA source.

# pick ECG affected channel
ch_idx = raw.ch_names.index('MEG 1531')
ecg = raw[ch_idx, :][0]

ecg = mne.filter.high_pass_filter(ecg.ravel(), raw.info['sfreq'], 1.0)

ecg_scores = ica.find_sources_raw(raw, target=ecg, score_func='pearsonr')

# get sources for the entire time range.
sources = ica.get_sources_raw(raw)

# get maximum correlation index for ECG
ecg_source_idx = np.abs(ecg_scores).argmax()

# high pass filter source
ecg_source = mne.filter.high_pass_filter(sources[ecg_source_idx],
                                         raw.info['sfreq'], 1.0)

###############################################################################
# Find ECG event onsets from ICA source
event_id = 999

ecg_events = ica_find_ecg_events(raw=raw, ecg_source=ecg_source,
                                 event_id=event_id)

# Read epochs
tmin, tmax = -0.1, 0.1
epochs = mne.Epochs(raw, ecg_events, event_id, tmin, tmax, picks=[ch_idx],
                    proj=False)

data = epochs.get_data()

print "Number of detected ECG artifacts : %d" % len(data)

###############################################################################
# Plot ECG artifacts
pl.figure()
pl.plot(1e3 * epochs.times, np.squeeze(data).T)
pl.xlabel('Times (ms)')
pl.ylabel('ECG')
pl.show()
