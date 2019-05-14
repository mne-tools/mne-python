# -*- coding: utf-8 -*-
"""
.. _tut-brainstorm-elekta-phantom:

==========================================
Brainstorm Elekta phantom dataset tutorial
==========================================

Here we compute the evoked from raw for the Brainstorm Elekta phantom
tutorial dataset. For comparison, see [1]_ and:

    https://neuroimage.usc.edu/brainstorm/Tutorials/PhantomElekta

References
----------
.. [1] Tadel F, Baillet S, Mosher JC, Pantazis D, Leahy RM.
       Brainstorm: A User-Friendly Application for MEG/EEG Analysis.
       Computational Intelligence and Neuroscience, vol. 2011, Article ID
       879716, 13 pages, 2011. doi:10.1155/2011/879716
"""
# sphinx_gallery_thumbnail_number = 9

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
data_path = bst_phantom_elekta.data_path(verbose=True)

raw_fname = op.join(data_path, 'kojak_all_200nAm_pp_no_chpi_no_ms_raw.fif')
raw = read_raw_fif(raw_fname)

###############################################################################
# Data channel array consisted of 204 MEG planor gradiometers,
# 102 axial magnetometers, and 3 stimulus channels. Let's get the events
# for the phantom, where each dipole (1-32) gets its own event:

events = find_events(raw, 'STI201')
raw.plot(events=events)
raw.info['bads'] = ['MEG1933', 'MEG2421']

###############################################################################
# The data have strong line frequency (60 Hz and harmonics) and cHPI coil
# noise (five peaks around 300 Hz). Here we plot only out to 60 seconds
# to save memory:

raw.plot_psd(tmax=30., average=False)

###############################################################################
# Our phantom produces sinusoidal bursts at 20 Hz:

raw.plot(events=events)

###############################################################################
# Now we epoch our data, average it, and look at the first dipole response.
# The first peak appears around 3 ms. Because we low-passed at 40 Hz,
# we can also decimate our data to save memory.

tmin, tmax = -0.1, 0.1
bmax = -0.05  # Avoid capture filter ringing into baseline
event_id = list(range(1, 33))
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, bmax),
                    preload=False)
epochs['1'].average().plot(time_unit='s')

###############################################################################
# .. _plt_brainstorm_phantom_elekta_eeg_sphere_geometry:
#
# Let's use a :ref:`sphere head geometry model <ch_forward_spherical_model>`
# and let's see the coordinate alignment and the sphere location. The phantom
# is properly modeled by a single-shell sphere with origin (0., 0., 0.).
sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=0.08)

mne.viz.plot_alignment(epochs.info, subject='sample', show_axes=True,
                       bem=sphere, dig=True, surfaces='inner_skull')

###############################################################################
# Let's do some dipole fits. We first compute the noise covariance,
# then do the fits for each event_id taking the time instant that maximizes
# the global field power.

# here we can get away with using method='oas' for speed (faster than "shrunk")
# but in general "shrunk" is usually better
cov = mne.compute_covariance(epochs, tmax=bmax)
mne.viz.plot_evoked_white(epochs['1'].average(), cov)

data = []
t_peak = 0.036  # true for Elekta phantom
for ii in event_id:
    # Avoid the first and last trials -- can contain dipole-switching artifacts
    evoked = epochs[str(ii)][1:-1].average().crop(t_peak, t_peak)
    data.append(evoked.data[:, 0])
evoked = mne.EvokedArray(np.array(data).T, evoked.info, tmin=0.)
del epochs
dip, residual = fit_dipole(evoked, cov, sphere, n_jobs=1)

###############################################################################
# Do a quick visualization of how much variance we explained, putting the
# data and residuals on the same scale (here the "time points" are the
# 32 dipole peak values that we fit):

fig, axes = plt.subplots(2, 1)
evoked.plot(axes=axes)
for ax in axes:
    ax.texts = []
    for line in ax.lines:
        line.set_color('#98df81')
residual.plot(axes=axes)

###############################################################################
# Now we can compare to the actual locations, taking the difference in mm:

actual_pos, actual_ori = mne.dipole.get_phantom_dipoles()
actual_amp = 100.  # nAm

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(6, 7))

diffs = 1000 * np.sqrt(np.sum((dip.pos - actual_pos) ** 2, axis=-1))
print('mean(position error) = %0.1f mm' % (np.mean(diffs),))
ax1.bar(event_id, diffs)
ax1.set_xlabel('Dipole index')
ax1.set_ylabel('Loc. error (mm)')

angles = np.rad2deg(np.arccos(np.abs(np.sum(dip.ori * actual_ori, axis=1))))
print(u'mean(angle error) = %0.1f°' % (np.mean(angles),))
ax2.bar(event_id, angles)
ax2.set_xlabel('Dipole index')
ax2.set_ylabel(u'Angle error (°)')

amps = actual_amp - dip.amplitude / 1e-9
print('mean(abs amplitude error) = %0.1f nAm' % (np.mean(np.abs(amps)),))
ax3.bar(event_id, amps)
ax3.set_xlabel('Dipole index')
ax3.set_ylabel('Amplitude error (nAm)')

fig.tight_layout()
plt.show()


###############################################################################
# Let's plot the positions and the orientations of the actual and the estimated
# dipoles

def plot_pos_ori(pos, ori, color=(0., 0., 0.), opacity=1.):
    """Plot dipole positions and orientations in 3D."""
    x, y, z = pos.T
    u, v, w = ori.T
    mlab.points3d(x, y, z, scale_factor=0.005, opacity=opacity, color=color)
    q = mlab.quiver3d(x, y, z, u, v, w,
                      scale_factor=0.03, opacity=opacity,
                      color=color, mode='arrow')
    q.glyph.glyph_source.glyph_source.shaft_radius = 0.02
    q.glyph.glyph_source.glyph_source.tip_length = 0.1
    q.glyph.glyph_source.glyph_source.tip_radius = 0.05


mne.viz.plot_alignment(evoked.info, bem=sphere, surfaces='inner_skull',
                       coord_frame='head', meg='helmet', show_axes=True)

# Plot the position and the orientation of the actual dipole
plot_pos_ori(actual_pos, actual_ori, color=(0., 0., 0.), opacity=0.5)
# Plot the position and the orientation of the estimated dipole
plot_pos_ori(dip.pos, dip.ori, color=(0.2, 1., 0.5))
mlab.view(70, 80, distance=0.5)
