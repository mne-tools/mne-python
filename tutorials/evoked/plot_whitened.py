"""
Plotting whitened data
======================

This tutorial demonstrates how to plot :term:`whitened <whitening>`
evoked data.

Data are whitened for many processes, including dipole fitting, source
localization and some decoding algorithms. Viewing whitened data thus gives
a different perspective on the data that these algorithms operate on.

Let's start by loading some data and computing a signal (spatial) covariance
that we'll consider to be noise.
"""

import mne
from mne.datasets import sample

###############################################################################
# Raw data with whitening
# -----------------------
# .. note:: In the :meth:`mne.io.Raw.plot` with ``noise_cov`` supplied,
#           you can press they "w" key to turn whitening on and off.

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)

events = mne.find_events(raw, stim_channel='STI 014')
event_id = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
            'visual/right': 4, 'smiley': 5, 'button': 32}
reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)
epochs = mne.Epochs(raw, events, event_id=event_id, reject=reject)

# baseline noise cov, not a lot of samples
noise_cov = mne.compute_covariance(epochs, tmax=0., method='shrunk', rank=None,
                                   verbose='error')

# butterfly mode shows the differences most clearly
raw.plot(events=events, butterfly=True)
raw.plot(noise_cov=noise_cov, events=events, butterfly=True)

###############################################################################
# Epochs with whitening
# ---------------------
epochs.plot()
epochs.plot(noise_cov=noise_cov)

###############################################################################
# Evoked data with whitening
# --------------------------

evoked = epochs.average()
evoked.plot(time_unit='s')
evoked.plot(noise_cov=noise_cov, time_unit='s')

###############################################################################
# Evoked data with scaled whitening
# ---------------------------------
# The :meth:`mne.Evoked.plot_white` function takes an additional step of
# scaling the whitened plots to show how well the assumption of Gaussian
# noise is satisfied by the data:

evoked.plot_white(noise_cov=noise_cov, time_unit='s')

###############################################################################
# Topographic plot with whitening
# -------------------------------

evoked.comment = 'All trials'
evoked.plot_topo(title='Evoked data')
evoked.plot_topo(noise_cov=noise_cov, title='Whitened evoked data')
