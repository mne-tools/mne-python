"""
==================================
Compute ICA components on Raw data
==================================

ICA is used to decompose raw data in 49 to 50 sources.
The source matching the ECG is found automatically
and displayed. Subsequently, the cleaned data is compared
with the uncleaned data. The last section shows how to export
the sources into a fiff file for further processing and displaying, e.g.
using mne_browse_raw.

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
from mne.preprocessing.ica import ICA
from mne.datasets import sample

###############################################################################
# Setup paths and prepare raw data

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

ica = ICA(n_components=0.90, max_n_components=100, noise_cov=None,
          random_state=0)
print ica

# 1 minute exposure should be sufficient for artifact detection.
# However, rejection performance may significantly improve when using
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
# Automatically find the ECG component using correlation with ECG signal.

# First, we create a helper function that iteratively applies the pearson
# correlation function to sources and returns an array of r values
# This is to illustrate the way ica.find_sources_raw works. Actually, this is
# the default score_func.

from scipy.stats import pearsonr

corr = lambda x, y: np.array([pearsonr(a, y.ravel()) for a in x])[:, 0]

# As we don't have an ECG channel we use one that correlates a lot with heart
# beats: 'MEG 1531'. We can directly pass the name to the find_sources method.
# In our example, the find_sources method returns and array of correlation
# scores for each ICA source.

ecg_scores = ica.find_sources_raw(raw, target='MEG 1531', score_func=corr)

# get sources
sources = ica.get_sources_raw(raw, start=start_plot, stop=stop_plot)

# get times
times = raw.time_as_index(np.arange(stop_plot - start_plot))

# get maximum correlation index for ECG
ecg_source_idx = np.abs(ecg_scores).argmax()

pl.figure()
pl.plot(times, sources[ecg_source_idx])
pl.title('ICA source matching ECG')
pl.show()

# let us have a look which other components resemble the ECG.
# We can do this by reordering the plot by our scores using order
# and generating sort indices for the sources:

ecg_order = np.abs(ecg_scores).argsort()[::-1]  # ascending order

ica.plot_sources_raw(raw, order=ecg_order, start=start_plot, stop=stop_plot)

# Let's make our ECG component selection more liberal and include sources
# for which the variance explanation in terms of \{r^2}\ exceeds 5 percent.

ecg_source_idx_updated = np.where(np.abs(ecg_scores) ** 2 > .05)[0]

###############################################################################
# Automatically find the EOG component using correlation with EOG signal.

# As we have an EOG channel, we can use it to detect the source.

eog_scores = ica.find_sources_raw(raw, target='EOG 061',
                                         score_func=corr)

# get maximum correlation index for EOG
eog_source_idx = np.abs(eog_scores).argmax()

# plot the component that correlates most with the EOG
pl.figure()
pl.plot(times, sources[eog_source_idx])
pl.title('ICA source matching EOG')
pl.show()

###############################################################################
# Show MEG data before and after ICA cleaning.

# Join the detected artifact indices.
exclude = np.r_[ecg_source_idx, eog_source_idx]

# Restore sources, use 64 PCA components which include the ICA cleaned sources
# plus additional PCA components not supplied to ICA (up to rank 64).
# This allows to control the trade-off between denoising and preserving data.
raw_ica = ica.pick_sources_raw(raw, include=None, exclude=exclude,
                               n_pca_components=64, copy=True)

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

###############################################################################
# Compare the affected channel before and after ICA cleaning.

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

###############################################################################
# Export ICA as raw for subsequent processing steps in ICA space.

from mne.layouts import make_grid_layout

ica_raw = ica.export_sources(raw, start=start, stop=stop, picks=None)

print ica_raw.ch_names

ica_lout = make_grid_layout(ica_raw.info)

# Uncomment the following two lines to save sources and layout.
# ica_raw.save('ica_raw.fif')
# ica_lout.save(os.path.join(os.environ['HOME'], '.mne/lout/ica.lout'))

################################################################################
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
