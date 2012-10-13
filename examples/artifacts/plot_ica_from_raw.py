"""
==================================
Compute ICA components on Raw data
==================================

"""
# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)


print __doc__

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

# 1 minute exposure should be sufficient for artifact detection
# however rejection pefromance significantly improves when using
# the entire data range
start, stop = raw.time_to_index(100, 160)

# setup ica seed
ica = ICA(noise_cov=None, n_components=25, random_state=0)
print ica

# decompose sources for raw data
ica.decompose_raw(raw, start=None, stop=None, picks=picks)
sources = ica.get_sources_raw(raw, picks=picks, start=start, stop=stop)

# # setup reasonable time window for inspection
start_plot, stop_plot = raw.time_to_index(100, 103)

# # plot components
plot_ica_panel(sources, start=0, stop=stop_plot - start_plot, n_components=25)


# TODO example broke somehow...
# Find the component that correlates the most with the ECG channel
# As we don't have an ECG channel with take one can correlates a lot
# 'MEG 1531'
# ecg, times = raw[raw.ch_names.index('MEG 1531'), start:stop]
# ecg = mne.filter.high_pass_filter(ecg.ravel(), raw.info['sfreq'], 1.)
# sources_corr = sources.copy()
# sources_corr /= np.sqrt(np.sum(sources_corr ** 2, axis=1))[:, np.newaxis]
# ecg_component_idx = np.argmax(np.dot(sources_corr, ecg.T))


# pl.plot(times, ica.sources[ecg_component_idx])
# pl.title('ICA source matching ECG')
# pl.show()

# Sign and order of components is non deterministic.
# however a distinct cardiac and one EOG component should be visible.

# raw_ica = ica.denoise_raw(bads=[ecg_component_idx], copy=True)
raw_ica = ica.pick_sources_raw(raw, bads=[0, 1], copy=True)


###############################################################################
# Show MEG data

data, times = raw[picks, start_plot:stop_plot]
ica_data, _ = raw_ica[picks, start_plot:stop_plot]

pl.close('all')
pl.plot(times, data.T)
pl.xlabel('time (s)')
pl.xlim(100, 103)
pl.ylabel('Raw MEG data (T)')

pl.figure()
pl.plot(times, ica_data.T)
pl.xlabel('time (s)')
pl.xlim(100, 103)
pl.ylabel('Denoised MEG data (T)')
pl.show()

# tmin, tmax, event_id = -0.2, 0.5, 1
# baseline = (None, 0)
# reject = None  # no artifact rejection besides ICA

# events = mne.find_events(raw, stim_channel='STI 014')

# epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
#                     baseline=baseline, preload=False, reject=reject)
# pl.figure()
# epochs.average().plot()

# epochs_ica = mne.Epochs(raw_ica, events, event_id, tmin, tmax, proj=True,
#                     picks=picks, baseline=baseline, preload=False,
#                     reject=reject)
# pl.figure()
# epochs_ica.average().plot()
