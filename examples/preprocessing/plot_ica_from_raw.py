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
from mne.fiff import Raw
from mne.artifacts.ica import ICA
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

# 1 minute exposure should be sufficient for artifact detection
# however rejection pefromance significantly improves when using
# the entire data range
start, stop = raw.time_as_index([100, 160])

# decompose sources for raw data
ica.decompose_raw(raw, start=start, stop=stop, picks=picks)
sources = ica.get_sources_raw(raw, start=start, stop=stop)

# setup reasonable time window for inspection
start_plot, stop_plot = raw.time_as_index([100, 103])

# plot components
ica.plot_sources_raw(raw, start=start_plot, stop=stop_plot)

###############################################################################
# Automatically find the ECG component using correlation with ECG signal

# First, we create a helper function that iteratively applies the pearson
# correlation functoon to sources and returns an array of r values
# This is to illustrate the way ica.find_sources_raw works. Actually, this is
# the default score_func.

from scipy.stats import pearsonr

corr = lambda x, y: np.array([pearsonr(a, y.ravel()) for a in x])[:, 0]

# As we don't have an ECG channel we use one that correlates a lot with heart
# beats: 'MEG 1531'. We can directly pass the name to the find_sources method.
# The default settings will take select the absolute maximum correlation.
# Thus, strongly negatively correlated sources will be found. (Another useful
# option would be to pass as additional argument select='all' which leads to
# all indices being returned, hence, allowing for custom threshold selection.
# For instance: source_idx[scores > .6]).

ecg_source_idx, _ = ica.find_sources_raw(raw, target='MEG 1531',
                                         score_func=corr)

sources = ica.get_sources_raw(raw, start=start_plot, stop=stop_plot)

times = raw.time_as_index(np.arange(stop_plot - start_plot))

pl.figure()
pl.plot(times, sources[ecg_source_idx].T)
pl.title('ICA source matching ECG')
pl.show()

###############################################################################
# Automatically find the EOG component using correlatio with EOG signal

# As we have an EOG channel, we can use it to detect the source.

eog_source_idx, _ = ica.find_sources_raw(raw, target='eog', score_func=corr)

# plot the component that correlates most with the EOG

pl.figure()
pl.plot(times, sources[eog_source_idx])
pl.title('ICA source matching EOG')
pl.show()

###############################################################################
# Show MEG data before and after ICA cleaning

exclude = np.r_[ecg_source_idx, eog_source_idx]

raw_ica = ica.pick_sources_raw(raw, include=None, exclude=exclude, copy=True)

start_compare, stop_compare = raw.time_as_index([100, 106])

data, times = raw[picks, start_compare:stop_compare]
ica_data, _ = raw_ica[picks, start_compare:stop_compare]

pl.figure()
pl.plot(times, data.T)
pl.xlabel('time (s)')
pl.xlim(100, 106)
pl.ylabel('Raw MEG data (T)')
y0, y1 = pl.ylim()

pl.figure()
pl.plot(times, ica_data.T)
pl.xlabel('time (s)')
pl.xlim(100, 106)
pl.ylabel('Denoised MEG data (T)')
pl.ylim(y0, y1)
pl.show()

################################################################################
# Compare the affected channel before and after ICA cleaning

affected_idx = raw.ch_names.index('MEG 1531')

# plot the component that correlates most with the ECG
pl.figure()
pl.plot(times, data[affected_idx])
pl.title('Affected channel MEG 1531 before cleaning.')
y0, y1 = pl.ylim()


# plot the component that correlates most with the ECG
pl.figure()
pl.plot(times, ica_data[affected_idx])
pl.title('Affected channel MEG 1531 after cleaning.')
pl.ylim(y0, y1)
pl.show()
