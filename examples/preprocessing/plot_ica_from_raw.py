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
from mne.viz import plot_ica_panel
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
plot_ica_panel(sources, start=0, stop=stop_plot - start_plot)

# Find the component that correlates the most with the ECG channel
# As we don't have an ECG channel with take one can correlates a lot
# 'MEG 1531'
affected_idx = raw.ch_names.index('MEG 1531')
ecg, times = raw[affected_idx, start:stop]
ecg = mne.filter.high_pass_filter(ecg.ravel(), raw.info['sfreq'], 1.)
sources_corr = sources.copy()
sources_corr /= np.sqrt(np.sum(sources_corr ** 2, axis=1))[:, np.newaxis]
ecg_component_idx = np.argmax(np.dot(sources_corr, ecg.T))

# plot the component that correlates most with the ecg
pl.figure()
pl.plot(times, sources[ecg_component_idx])
pl.title('ICA source matching ECG')

# In addition a distinct cardiac and one EOG component
# should be visible (0 and 1).
raw_ica = ica.pick_sources_raw(raw, include=None,
                               exclude=[ecg_component_idx, 0, 1], copy=True)

###############################################################################
# Show MEG data

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
# Compare the affected channel before and after ICA cleaning/

# plot the component that correlates most with the ecg
pl.figure()
pl.plot(times, data[affected_idx])
pl.title('Affected channel MEG 1531 before cleaning.')
y0, y1 = pl.ylim()


# plot the component that correlates most with the ecg
pl.figure()
pl.plot(times, ica_data[affected_idx])
pl.title('Affected channel MEG 1531 after cleaning.')
pl.ylim(y0, y1)
pl.show()
