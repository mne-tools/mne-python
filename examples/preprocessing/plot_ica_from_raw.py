"""
==================================
Compute ICA components on raw data
==================================

ICA is used to decompose raw data in 49 to 50 sources.
The source matching the ECG is found automatically
and displayed. Subsequently, the cleaned data is compared
with the uncleaned data. The last section shows how to export
the sources into a fiff file for further processing and displaying, e.g.
using mne_browse_raw.

"""
print(__doc__)

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.io import Raw
from mne.preprocessing.ica import ICA
from mne.datasets import sample
from mne.filter import band_pass_filter

###############################################################################
# Setup paths and prepare raw data

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = Raw(raw_fname, preload=True)

raw.filter(1, 45, n_jobs=2)

picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                       stim=False, exclude='bads')

###############################################################################
# Setup ICA seed decompose data, then access and plot sources.

# Instead of the actual number of components here we pass a float value
# between 0 and 1 to select n_components based on the percentage of
# variance explained by the PCA components.

ica = ICA(n_components=0.90, n_pca_components=None, max_pca_components=None,
          random_state=0)

# Also we decide to use all PCA components before mixing back to sensor space.
# You can again use percentages (float) or set the total number of components
# to be kept directly (int) which allows to control the amount of additional
# denoising.

ica.n_pca_components = 1.0

# decompose sources for raw data using each third sample.
ica.decompose_raw(raw, picks=picks, decim=3)
print(ica)

# plot reasonable time window for inspection
start_plot, stop_plot = 100., 103.
ica.plot_sources_raw(raw, range(30), start=start_plot, stop=stop_plot)

###############################################################################
# Automatically find the ECG component using correlation with ECG signal.

# Defining a customized distance function.

# You can pass any function object that
# takes a n_sources X n_samples vector and, optionally, a second
# n_samples vector, and returns a score vector of length n_sources.
# Let's illustrate this by creating a function that, when passed as
# `score_func` argument, does the same as the default value 'pearsonr'.

from scipy.stats import pearsonr


def score_func(x, y):
    """Return pearson correlation between ICA source
       and target time series
    """
    correlation, pval = np.array([pearsonr(a, y) for a in x]).T
    return correlation

# As we don't have an ECG channel we use one that correlates a lot with heart
# beats: 'MEG 1531'. To improve detection, we filter the the channel and pass
# it directly to find sources. The method then returns an array of correlation
# scores for each ICA source.

ecg_ch_name = 'MEG 1531'
l_freq, h_freq = 8, 16
ecg = raw[[raw.ch_names.index(ecg_ch_name)], :][0]
ecg = band_pass_filter(ecg, raw.info['sfreq'], l_freq, h_freq)
ecg_scores = ica.find_sources_raw(raw, target=ecg, score_func=score_func)

# get maximum correlation index for ECG
ecg_source_idx = np.abs(ecg_scores).argmax()
title = 'ICA source matching ECG'
ica.plot_sources_raw(raw, ecg_source_idx, title=title, stop=3.0)

# let us have a look which other components resemble the ECG.
# We can do this by reordering the plot by our scores using order
# and generating sort indices for the sources:

ecg_order = np.abs(ecg_scores).argsort()[::-1]  # ascending order

ica.plot_sources_raw(raw, ecg_order[:15], start=start_plot, stop=stop_plot)

ica.plot_topomap(ecg_order[:15], colorbar=False)

ecg_inds = np.abs(ecg_scores).argsort()[-3:]  # take the first 3 components

# visualize scores
ica.plot_scores(ecg_scores, exclude=ecg_inds, title='correlation with ECG')

ica.exclude.extend(ecg_inds)


###############################################################################
# Automatically find the EOG component using correlation with EOG signal.

# As we have an EOG channel, we can use it to detect the source.

eog_scores = ica.find_sources_raw(raw, target='EOG 061', score_func=score_func)

# get maximum correlation index for EOG
eog_source_idx = np.abs(eog_scores).argmax()

# plot the component that correlates most with the EOG
title = 'ICA source matching EOG'
ica.plot_sources_raw(raw, eog_source_idx, title=title, stop=3.0)

# plot spatial sensitivities of EOG and ECG ICA components
title = 'Spatial patterns of ICA components for ECG+EOG (Magnetometers)'
source_idx = range(15)
ica.plot_topomap([ecg_source_idx, eog_source_idx], ch_type='mag')
plt.suptitle(title, fontsize=12)

###############################################################################
# Show MEG data before and after ICA cleaning.

# We now add the eog artifacts to the ica.exclusion list
ica.exclude += [eog_source_idx]

# Restore sensor space data and keep all PCA components
raw_ica = ica.pick_sources_raw(raw, include=None, n_pca_components=1.0)

# let's now compare the date before and after cleaning.
start_compare, stop_compare = raw.time_as_index([100, 106])
data, times = raw[picks, start_compare:stop_compare]
data_clean, _ = raw_ica[picks, start_compare:stop_compare]

# first the raw data
plt.figure()
plt.plot(times, data.T, color='r')
plt.plot(times, data_clean.T, color='k')
plt.xlabel('time (s)')
plt.xlim(100, 106)
plt.show()

# now the affected channel
affected_idx = raw.ch_names.index('MEG 1531')
plt.figure()
plt.plot(times, data[affected_idx], color='r')
plt.plot(times, data_clean[affected_idx], color='k')
plt.xlim(100, 106)
plt.show()


###############################################################################
# Validation: check ECG components extracted

# Export ICA as Raw object for subsequent processing steps in ICA space.

ica_raw = ica.sources_as_raw(raw, start=100., stop=160., picks=None)

from mne.preprocessing import find_ecg_events

# find ECG events
event_id = 999
events, _, _ = find_ecg_events(raw, ch_name='MEG 1531', event_id=event_id,
                               l_freq=8, h_freq=16)

# pick components, create epochs and evoked in ICA space
ica_picks = np.arange(ica.n_components_)

ica_raw.info['bads'] = []  # selected components are exported as bad channels

# create epochs around ECG events
ecg_epochs = mne.Epochs(ica_raw, events=events, event_id=event_id,
                        tmin=-0.5, tmax=0.5, baseline=None, proj=False,
                        picks=ica_picks)

ica_ave = ecg_epochs.average(ica_picks)

plt.figure()
times = ica_ave.times * 1e3

plt.plot(times, ica_ave.data.T, 'k')  # plot unclassified sources
for ii in ecg_inds:  # use indexing to expose ECG related sources
    color, label = ('r', 'ICA %02d' % ii)
    plt.plot(times, ica_ave.data[ii], color=color, label=label)

plt.xlim(times[[0, -1]])
plt.legend()
plt.show()

###############################################################################
# To save an ICA session you can say:
# ica.save('my_ica.fif')
#
# You can later restore the session by saying:
# >>> from mne.preprocessing import read_ica
# >>> read_ica('my_ica.fif')
#
# The ICA functionality exposed in this example will then be available at
# at any later point in time provided the data have the same structure as the
# data initially supplied to ICA.
