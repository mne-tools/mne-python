"""
================================
Compute ICA components on Epochs
================================

25 ICA components are estimated and displayed.

"""
print __doc__

# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import matplotlib.pylab as pl
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
ica = ICA(noise_cov=None, n_components=25, random_state=0)
print ica

# get epochs
tmin, tmax, event_id = -0.2, 0.5, 1
# baseline = None
baseline = (None, 0)
reject = None

events = mne.find_events(raw, stim_channel='STI 014')
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=baseline, preload=True, reject=reject)


# fit sources from epochs or from raw (both works for epochs)
ica.decompose_epochs(epochs)

# get sources from epochs
sources = ica.get_sources_epochs(epochs)

# plot components for one epoch of interest
plot_ica_panel(sources[13], n_components=25)

# A distinct cardiac component should be visible at index 24
epochs_ica = ica.pick_sources_epochs(epochs, include=None, exclude=[24],
                                     copy=True)

# plot original epochs
pl.figure()
epochs.average().plot()
pl.show()

# plot cleaned epochs
pl.figure()
epochs_ica.average().plot()
pl.show()
