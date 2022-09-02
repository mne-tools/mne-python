# -*- coding: utf-8 -*-
"""
.. _tut-brainstorm-elekta-phantom:

==========================================
Brainstorm Elekta phantom dataset tutorial
==========================================

Here we compute the evoked from raw for the Brainstorm Elekta phantom
tutorial dataset. For comparison, see :footcite:`TadelEtAl2011` and
`the original Brainstorm tutorial
<https://neuroimage.usc.edu/brainstorm/Tutorials/PhantomElekta>`__.
"""
# sphinx_gallery_thumbnail_number = 9

# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

# %%

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import find_events, fit_dipole
from mne.datasets import fetch_phantom
from mne.datasets.brainstorm import bst_phantom_elekta
from mne.io import read_raw_fif

print(__doc__)

# %%
# The data were collected with an Elekta Neuromag VectorView system at 1000 Hz
# and low-pass filtered at 330 Hz. Here the medium-amplitude (200 nAm) data
# are read to construct instances of :class:`mne.io.Raw`.
data_path = bst_phantom_elekta.data_path(verbose=True)

raw_fname = data_path / 'kojak_all_200nAm_pp_no_chpi_no_ms_raw.fif'
raw = read_raw_fif(raw_fname)

# %%
# Data channel array consisted of 204 MEG planor gradiometers,
# 102 axial magnetometers, and 3 stimulus channels. Let's get the events
# for the phantom, where each dipole (1-32) gets its own event:

events = find_events(raw, 'STI201')
raw.plot(events=events)
raw.info['bads'] = ['MEG1933', 'MEG2421']

# %%
# The data have strong line frequency (60 Hz and harmonics) and cHPI coil
# noise (five peaks around 300 Hz). Here we plot only out to 60 seconds
# to save memory:

raw.plot_psd(tmax=30., average=False)

# %%
# Our phantom produces sinusoidal bursts at 20 Hz:

raw.plot(events=events)

# %%
# Now we epoch our data, average it, and look at the first dipole response.
# The first peak appears around 3 ms. Because we low-passed at 40 Hz,
# we can also decimate our data to save memory.

tmin, tmax = -0.1, 0.1
bmax = -0.05  # Avoid capture filter ringing into baseline
event_id = list(range(1, 33))
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, bmax),
                    preload=False)
epochs['1'].average().plot(time_unit='s')

# %%
# .. _plt_brainstorm_phantom_elekta_eeg_sphere_geometry:
#
# Let's use a :ref:`sphere head geometry model <eeg_sphere_model>`
# and let's see the coordinate alignment and the sphere location. The phantom
# is properly modeled by a single-shell sphere with origin (0., 0., 0.).
#
# Even though this is a VectorView/TRIUX phantom, we can use the Otaniemi
# phantom subject as a surrogate because the "head" surface (hemisphere outer
# shell) has the same geometry for both phantoms, even though the internal
# dipole locations differ. The phantom_otaniemi scan was aligned to the
# phantom's head coordinate frame, so an identity ``trans`` is appropriate
# here.

subjects_dir = data_path
fetch_phantom('otaniemi', subjects_dir=subjects_dir)
sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=0.08)
subject = 'phantom_otaniemi'
trans = mne.transforms.Transform('head', 'mri', np.eye(4))
mne.viz.plot_alignment(
    epochs.info, subject=subject, show_axes=True, bem=sphere, dig=True,
    surfaces=('head-dense', 'inner_skull'), trans=trans, mri_fiducials=True,
    subjects_dir=subjects_dir)

# %%
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
dip, residual = fit_dipole(evoked, cov, sphere, n_jobs=None)

# %%
# Do a quick visualization of how much variance we explained, putting the
# data and residuals on the same scale (here the "time points" are the
# 32 dipole peak values that we fit):

fig, axes = plt.subplots(2, 1)
evoked.plot(axes=axes)
for ax in axes:
    for text in list(ax.texts):
        text.remove()
    for line in ax.lines:
        line.set_color('#98df81')
residual.plot(axes=axes)

# %%
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

# %%
# Let's plot the positions and the orientations of the actual and the estimated
# dipoles

actual_amp = np.ones(len(dip))  # misc amp to create Dipole instance
actual_gof = np.ones(len(dip))  # misc GOF to create Dipole instance
dip_true = \
    mne.Dipole(dip.times, actual_pos, actual_amp, actual_ori, actual_gof)

fig = mne.viz.plot_alignment(
    evoked.info, trans, subject, bem=sphere, surfaces={'head-dense': 0.2},
    coord_frame='head', meg='helmet', show_axes=True,
    subjects_dir=subjects_dir)

# Plot the position and the orientation of the actual dipole
fig = mne.viz.plot_dipole_locations(dipoles=dip_true, mode='arrow',
                                    subject=subject, color=(0., 0., 0.),
                                    fig=fig)

# Plot the position and the orientation of the estimated dipole
fig = mne.viz.plot_dipole_locations(dipoles=dip, mode='arrow', subject=subject,
                                    color=(0.2, 1., 0.5), fig=fig)

mne.viz.set_3d_view(figure=fig, azimuth=70, elevation=80, distance=0.5)

# %%
# References
# ----------
# .. footbibliography::
