"""
==================================
Compute ICA components on Raw data
==================================

"""
# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

print __doc__

import matplotlib.pylab as pl
import mne
from mne.fiff import Raw
# from mne.artifacts.ica import decompose_raw
from mne.artifacts.ica import ICA
from mne.viz import plot_ica_panel

from mne.datasets import sample
data_path = sample.data_path('..')
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = Raw(raw_fname, preload=True)

picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, eog=False,
                            stim=False, exclude=raw.info['bads'])


start, stop = raw.time_to_index(100, 160)

# setup ica seed
ica = ICA(noise_cov=None, n_components=25, random_state=0)
print ica

# get epochs
tmin, tmax, event_id = -0.2, 0.5, 1
baseline = (None, 0)
reject = None
# reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)

events = mne.find_events(raw, stim_channel='STI 014')

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=baseline, preload=False, reject=reject)

# fit sources from epochs or from raw (both works for epochs)
ica.decompose_epochs(epochs, picks=picks)

# get sources from epochs
sources = ica.get_sources_epochs(epochs)

# setup reasonable time window for inspection
start_plot, stop_plot = 0, 1000

# plot components for epoch of interest
plot_ica_panel(sources[13], start=start_plot, stop=stop_plot, n_components=25)

# A distinct cardiac component should be visible
epochs_ica = ica.pick_sources_epochs(epochs, exclude=[24], copy=True)

# plot original epochs
pl.figure()
epochs.average().plot()
pl.show()

# plot cleaned epochs
pl.figure()
epochs_ica.average().plot()
pl.show()
