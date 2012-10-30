"""
==================================
Compute ICA components on Raw data
==================================

ICA is used to decompose raw data in 25 sources.
The source matching the ECG is found automatically
and displayed.

"""
print __doc__

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import numpy as np
import pylab as pl

import mne
from mne.fiff import Raw, pick_types
from mne.artifacts.ica import ICA, ica_find_eog_events
from mne.datasets import sample

data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = Raw(raw_fname, preload=True)

picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, eog=False,
                            stim=False, exclude=raw.info['bads'])

# setup ica seed
# Sign and order of components is non deterministic.
# setting the random state to 0 helps stabilizing the solution.
ica = ICA(noise_cov=None, n_components=25, random_state=0)
print ica

# 1 minute exposure should be sufficient for artifact detection.
# However, rejection pefromance may significantly improve when using
# the entire data range
start, stop = raw.time_as_index([100, 160])

# decompose sources for raw data
ica.decompose_raw(raw, start=start, stop=stop, picks=picks)
print ica

sources = ica.get_sources_raw(raw, start=start, stop=stop)

# setup reasonable time window for inspection
start_plot, stop_plot = raw.time_as_index([100, 103])

# plot components
ica.plot_sources_raw(raw, start=start_plot, stop=stop_plot)

###############################################################################
# Automatically find the ECG component using correlation with ECG signal

# As we don't have an ECG channel we use one that correlates a lot with heart
# beats: 'MEG 1531'. We can directly pass the name to the find_sources method.
# We select the pearson correlation from scipy stats via string lable.
# The function is internally modified to be applicable to 2D arrays and,
# hence, returns product-moment correlation scores for each ICA source.

eog_scores = ica.find_sources_raw(raw, target='EOG 061',
                                  score_func='pearsonr')

# get sources for the entire time range.
sources = ica.get_sources_raw(raw)

# get maximum correlation index for ECG
eog_source_idx = np.abs(eog_scores).argmax()

###############################################################################
# Find ECG event onsets from ICA source
event_id = 999

eog_events = ica_find_eog_events(raw=raw, eog_source=sources[eog_source_idx],
                                 event_id=event_id)

# Read epochs
picks = pick_types(raw.info, meg=False, eeg=False, stim=False, eog=False,
                   include=['EOG 061'])

tmin, tmax = -0.2, 0.2
epochs = mne.Epochs(raw, eog_events, event_id, tmin, tmax, picks=picks,
                    proj=False)

data = epochs.get_data()

print "Number of detected EOG artifacts : %d" % len(data)

###############################################################################
# Plot EOG artifacts
pl.figure()
pl.plot(1e3 * epochs.times, np.squeeze(data).T)
pl.xlabel('Times (ms)')
pl.ylabel('EOG')
pl.show()
