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
import matplotlib.pyplot as plt

import mne
from mne import find_events, fit_dipole
from mne.datasets.brainstorm import bst_phantom_elekta
from mne.io import read_raw_fif

from mayavi import mlab
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

raw.plot_psd(tmax=60., average=False)

###############################################################################
# Let's use Maxwell filtering to clean the data a bit.
# Ideally we would have the fine calibration and cross-talk information
# for the site of interest, but we don't, so we just do:

raw.fix_mag_coil_types()
raw = mne.preprocessing.maxwell_filter(raw, origin=(0., 0., 0.))

###############################################################################
# We know our phantom produces sinusoidal bursts below 25 Hz, so let's filter.

raw.filter(None, 40., fir_design='firwin')
raw.plot(events=events)

###############################################################################
# Now we epoch our data, average it, and look at the first dipole response.
# The first peak appears around 3 ms. Because we low-passed at 40 Hz,
# we can also decimate our data to save memory.

tmin, tmax = -0.1, 0.1
event_id = list(range(1, 33))
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, -0.01),
                    decim=3, preload=True)
epochs['1'].average().plot()

###############################################################################
# Let's use a sphere head geometry model and let's see the coordinate
# alignement and the sphere location. The phantom is properly modeled by
# a single-shell sphere with origin (0., 0., 0.).
sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=None)

mne.viz.plot_alignment(raw.info, subject='sample',
                       meg='helmet', bem=sphere, dig=True,
                       surfaces=['brain'])

###############################################################################
# Let's do some dipole fits. We first compute the noise covariance,
# then do the fits for each event_id taking the time instant that maximizes
# the global field power.

cov = mne.compute_covariance(epochs, tmax=0)
data = []
for ii in event_id:
    evoked = epochs[str(ii)].average()
    idx_peak = np.argmax(evoked.copy().pick_types(meg='grad').data.std(axis=0))
    t_peak = evoked.times[idx_peak]
    evoked.crop(t_peak, t_peak)
    data.append(evoked.data[:, 0])
evoked = mne.EvokedArray(np.array(data).T, evoked.info, tmin=0.)
del epochs, raw
dip = fit_dipole(evoked, cov, sphere, n_jobs=1)[0]

###############################################################################
# Now we can compare to the actual locations, taking the difference in mm:

actual_pos, actual_ori = mne.dipole.get_phantom_dipoles()
actual_amp = 100.  # nAm

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(6, 7))

diffs = 1000 * np.sqrt(np.sum((dip.pos - actual_pos) ** 2, axis=-1))
print('mean(position error) = %s' % (np.mean(diffs),))
ax1.bar(event_id, diffs)
ax1.set_xlabel('Dipole index')
ax1.set_ylabel('Loc. error (mm)')

angles = np.arccos(np.abs(np.sum(dip.ori * actual_ori, axis=1)))
print('mean(angle error) = %s' % (np.mean(angles),))
ax2.bar(event_id, angles)
ax2.set_xlabel('Dipole index')
ax2.set_ylabel('Angle error (rad)')

amps = actual_amp - dip.amplitude / 1e-9
print('mean(abs amplitude error) = %s' % (np.mean(np.abs(amps)),))
ax3.bar(event_id, amps)
ax3.set_xlabel('Dipole index')
ax3.set_ylabel('Amplitude error (nAm)')

fig.tight_layout()
plt.show()


###############################################################################
# Let's plot the positions and the orientations of the actual and the estimated
# dipoles
def plot_pos_ori(pos, ori, color=(0., 0., 0.)):
    mlab.points3d(pos[:, 0], pos[:, 1], pos[:, 2], scale_factor=0.005,
                  color=color)
    mlab.quiver3d(pos[:, 0], pos[:, 1], pos[:, 2],
                  ori[:, 0], ori[:, 1], ori[:, 2],
                  scale_factor=0.03,
                  color=color)

mne.viz.plot_alignment(evoked.info, bem=sphere, surfaces=[])

# Plot the position and the orientation of the actual dipole
plot_pos_ori(actual_pos, actual_ori, color=(1., 0., 0.))
# Plot the position and the orientation of the estimated dipole
plot_pos_ori(dip.pos, dip.ori, color=(0., 0., 1.))
