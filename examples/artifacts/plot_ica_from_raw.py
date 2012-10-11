"""
==================================
Compute ICA components on Raw data
==================================

"""
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
start, stop = raw.time_to_index(100, 160)

# setup ica seed
ica = ICA(noise_cov=None, n_components=25, random_state=0)
print ica

# fit sources for raw data
ica.fit_raw(raw, picks, start=start, stop=stop)

# Find the component that correlates the most with the ECG channel
# As we don't have an ECG channel with take one can correlates a lot
# 'MEG 1531'
ecg, times = raw[raw.ch_names.index('MEG 1531'), start:stop]
ecg = mne.filter.high_pass_filter(ecg.ravel(), raw.info['sfreq'], 1.)
sources = ica.sources.copy()
sources /= np.sqrt(np.sum(sources ** 2, axis=1))[:, np.newaxis]
ecg_component_idx = np.argmax(np.dot(sources, ecg.T))

import pylab as pl
pl.plot(times, ica.sources[ecg_component_idx])
pl.title('ICA source matching ECG')
pl.show()

# setup reasonable time window for inspection
start_plot, stop_plot = raw.time_to_index(100, 103)

# plot components
plot_ica_panel(ica, start=0, stop=stop_plot - start_plot)

# Sign and order of components is non deterministic.
# however a distinct cardiac and one EOG component should be visible.

raw_ica = ica.denoise_raw(bads=[ecg_component_idx], copy=True)

###############################################################################
# Show MEG data
some_picks = picks[:5]  # take 5 first
data, times = raw[some_picks, start:(stop + 1)]
ica_data, _ = raw[some_picks, start:(stop + 1)]

import pylab as pl
pl.close('all')
pl.plot(times, data.T)
pl.xlabel('time (s)')
pl.ylabel('Raw MEG data (T)')

pl.figure()
pl.plot(times, ica_data.T)
pl.xlabel('time (s)')
pl.ylabel('Denoised MEG data (T)')
pl.show()

# tmin, tmax, event_id = -0.2, 0.5, 1
# baseline = (None, 0)
# reject = None  # no artifact rejection besides ICA
# 
# events = mne.find_events(raw, stim_channel='STI 014')
# 
# epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
#                     baseline=baseline, preload=False, reject=reject)
# pl.figure()
# epochs.average().plot()
# 
# epochs_ica = mne.Epochs(raw_ica, events, event_id, tmin, tmax, proj=True,
#                     picks=picks, baseline=baseline, preload=False,
#                     reject=reject)
# pl.figure()
# epochs_ica.average().plot()
