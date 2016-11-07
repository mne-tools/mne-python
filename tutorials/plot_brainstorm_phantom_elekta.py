# -*- coding: utf-8 -*-
"""
==========================================
Brainstorm Elekta phantom tutorial dataset
==========================================

Here we compute the evoked from raw for the Brainstorm Elekta phantom
tutorial dataset. For comparison, see [1]_ and:

    http://neuroimage.usc.edu/brainstorm/Tutorials/PhantomElekta

References
----------
.. [1] Tadel F, Baillet S, Mosher JC, Pantazis D, Leahy RM.
       Brainstorm: A User-Friendly Application for MEG/EEG Analysis.
       Computational Intelligence and Neuroscience, vol. 2011, Article ID
       879716, 13 pages, 2011. doi:10.1155/2011/879716
"""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np

import mne
from mne import find_events, fit_dipole
from mne.datasets.brainstorm import bst_phantom_elekta
from mne.io import read_raw_fif

print(__doc__)

###############################################################################
# The data were collected with an Elekta Neuromag VectorView system at 1000 Hz
# and low-pass filtered at 330 Hz. Here the medium-amplitude (200 nAm) data
# are read to construct instances of :class:`mne.io.Raw`.
data_path = bst_phantom_elekta.data_path()

raw_fname = op.join(data_path, 'kojak_all_200nAm_pp_no_chpi_no_ms_raw.fif')
raw = read_raw_fif(raw_fname)

###############################################################################
# Data channel array consisted of 204 MEG planor gradiometers,
# 102 axial magnetometers, and 3 stimulus channels. Let's get the events
# for the phantom, where each dipole (1-32) gets its own event:

events = find_events(raw, 'STI201')
raw.plot(events=events)
raw.info['bads'] = ['MEG2421']

###############################################################################
# The data have strong line frequency (60 Hz and harmonics) and cHPI coil
# noise (five peaks around 300 Hz). Here we plot only out to 60 seconds
# to save memory:

raw.plot_psd(tmax=60.)

###############################################################################
# Let's use Maxwell filtering to clean the data a bit.
# Ideally we would have the fine calibration and cross-talk information
# for the site of interest, but we don't, so we just do:

raw.fix_mag_coil_types()
raw = mne.preprocessing.maxwell_filter(raw, origin=(0., 0., 0.))

###############################################################################
# We know our phantom produces sinusoidal bursts below 25 Hz, so let's filter.

raw.filter(None, 40., h_trans_bandwidth='auto', filter_length='auto',
           phase='zero', fir_window='hamming')
raw.plot(events=events)

###############################################################################
# Now we epoch our data, average it, and look at the first dipole response.
# The first peak appears around 3 ms. Because we low-passed at 40 Hz,
# we can also decimate our data to save memory.

tmin, tmax = -0.1, 0.1
event_id = list(range(1, 33))
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, -0.01),
                    decim=5, preload=True)
epochs['1'].average().plot()

###############################################################################
# Let's do some dipole fits. The phantom is properly modeled by a single-shell
# sphere with origin (0., 0., 0.). We compute covariance, then do the fits.

t_peak = 60e-3  # ~60 MS at largest peak
sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=None)
cov = mne.compute_covariance(epochs, tmax=0)
data = []
for ii in range(1, 33):
    evoked = epochs[str(ii)].average().crop(t_peak, t_peak)
    data.append(evoked.data[:, 0])
evoked = mne.EvokedArray(np.array(data).T, evoked.info, tmin=0.)
del epochs, raw
dip = fit_dipole(evoked, cov, sphere, n_jobs=1)[0]

###############################################################################
# Now we can compare to the actual locations, taking the difference in mm:

actual_pos = mne.dipole.get_phantom_dipoles()[0]
diffs = 1000 * np.sqrt(np.sum((dip.pos - actual_pos) ** 2, axis=-1))
print('Differences (mm):\n%s' % diffs[:, np.newaxis])
print('μ = %s' % (np.mean(diffs),))
