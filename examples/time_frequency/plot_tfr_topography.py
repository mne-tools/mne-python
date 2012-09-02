"""
===================================================================
Plot time-frequency representations on topographies for MEG sensors
===================================================================

"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

print __doc__

import pylab as pl

from mne import fiff
from mne.layouts import Layout
from mne.viz import plot_topo_tfr
import pylab as pl
from mne.datasets import sample
data_path = sample.data_path('..')

raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
event_id, tmin, tmax = 1, -0.2, 0.5

# Setup for reading the raw data
raw = fiff.Raw(raw_fname)
events = mne.read_events(event_fname)

include = []
exclude = raw.info['bads'] + ['MEG 2443', 'EEG 053']  # bads + 2 more

# picks MEG gradiometers
picks = fiff.pick_types(raw.info, meg='grad', eeg=False, eog=True,
                                stim=False, include=include, exclude=exclude)

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6))
data = epochs.get_data()  # as 3D matrix

###############################################################################
# Calculate power and phase_locking value

frequencies = np.arange(7, 30, 3)  # define frequencies of interest
n_cycles = frequencies / float(7)  # different number of cycle per frequency
Fs = raw.info['sfreq']  # sampling in Hz
decim = 3
power, phase_lock = induced_power(data, Fs=Fs, frequencies=frequencies,
                                  n_cycles=n_cycles, n_jobs=1, use_fft=False,
                                  decim=decim, zero_mean=True)

layout = Layout('Vectorview-all')

###############################################################################
# Show topography of power (pe patient, may this take some time)

plot_topo_tfr(epochs, power, frequencies, lauyout , is_power=True, decim=decim)
title = '%s - MNE sample data' %"Induced power"
pl.figtext(0.03, 0.93, title, color='w', fontsize=18)
pl.show()


###############################################################################
# Show topography of phase_locking value (pe patient, this may take some time)

plot_topo_tfr(epochs, phase_lock, frequencies, lauyout , is_power=False, decim=decim)
title = '%s - MNE sample data' %"Phase locking value"
pl.figtext(0.03, 0.93, title, color='w', fontsize=18)
pl.show()

