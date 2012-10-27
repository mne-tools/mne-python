"""
==================================
Compute ICA components on Raw data
==================================

ICA is used to decompose raw data in 25 sources.
Events are extracted from the ecg sources.

"""
print __doc__

# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import numpy as np

import mne
from mne.fiff import Raw
from mne.artifacts.ica import ICA
from mne.datasets import sample
from mne.fiff import pick_types

data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = Raw(raw_fname, preload=True)

picks = pick_types(raw.info, meg=True, eeg=False, eog=False,
                            stim=False, exclude=raw.info['bads'])

# setup ica seed
# Sign and order of components is non deterministic.
# setting the random state to 0 helps stabilizing the solution.
ica = ICA(noise_cov=None, n_components=25, random_state=0)
print ica

# 1 minute exposure should be sufficient for artifact detection
# however rejection pefromance significantly improves when using
# the entire data range
start, stop = raw.time_as_index([100, 160])

# decompose sources for raw data using the full length
ica.decompose_raw(raw, start=start, stop=stop, picks=picks)

# setup reasonable time window for inspection
start_plot, stop_plot = raw.time_as_index([100, 103])

# plot components
ica.plot_panel(raw, start=0, stop=(stop_plot - start_plot))

# Find the component that correlates the most with the ECG channel
# As we don't have an ECG channel with take one can correlates a lot
# 'MEG 1531'
affected_idx = raw.ch_names.index('MEG 1531')
ecg, times = raw[affected_idx]
ecg = mne.filter.high_pass_filter(ecg.ravel(), raw.info['sfreq'], 1.)

from scipy.stats import pearsonr

#  create function that iteratively applies the pearson correlation
#  to sources the sources and returns and array of r values
corr_r = lambda x, y: np.array([pearsonr(a, y.ravel()) for a in x])[:, 0]

# get source with maximum absolute correlation to the ecg signal
# as we don't have an
source_idx = ica.find_sources(raw, target='MEG 1531', score_func=corr_r,
                              criterion='max')

print source_idx

# get source with maximum positive correlation
source_idx = ica.find_sources(raw, target=ecg, score_func=corr_r,
                              take_abs=False, criterion='max')

print source_idx

# get source with maximum negative correlation
source_idx = ica.find_sources(raw, target=ecg, score_func=corr_r,
                              take_abs=False, criterion='min')

print source_idx


###############################################################################
#  get sources pearson correlated with ecg signal above a minimum variance
#  explanation value of 2 percent

#  create function that iteratively applies pearson correlation to sources
#  and the ecg and returns \{r}^2\
corr_expl = lambda x, y: np.power([pearsonr(a, y.ravel()) for a in x], 2)[:, 0]

# set cruterion to tuple with a comparison function to the left
# and the float criterion to the right
criterion = (np.greater_equal, .02)

source_idx = ica.find_sources(raw, target=ecg, score_func=corr_expl,
                              criterion=criterion)

print source_idx
