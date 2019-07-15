# -*- coding: utf-8 -*-
"""
============================================
4D Neuroimaging/BTi phantom dataset tutorial
============================================

Here we read 4DBTi epochs data obtained with a spherical phantom
using four different dipole locations. For each condition we
compute evoked data and compute dipole fits.

Data are provided by Jean-Michel Badier from MEG center in Marseille, France.
"""

# Authors: Alex Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np
from mayavi import mlab
from mne.datasets import phantom_4dbti
import mne

###############################################################################
# Read data and compute a dipole fit at the peak of the evoked response

data_path = phantom_4dbti.data_path()
raw_fname = op.join(data_path, '%d/e,rfhp1.0Hz')

dipoles = list()
sphere = mne.make_sphere_model(r0=(0., 0., 0.), head_radius=0.080)

t0 = 0.07  # peak of the response

pos = np.empty((4, 3))

for ii in range(4):
    raw = mne.io.read_raw_bti(raw_fname % (ii + 1,),
                              rename_channels=False, preload=True)
    raw.info['bads'] = ['A173', 'A213', 'A232']
    events = mne.find_events(raw, 'TRIGGER', mask=4350, mask_type='not_and')
    epochs = mne.Epochs(raw, events=events, event_id=8192, tmin=-0.2, tmax=0.4,
                        preload=True)
    evoked = epochs.average()
    evoked.plot(time_unit='s')
    cov = mne.compute_covariance(epochs, tmax=0.)
    dip = mne.fit_dipole(evoked.copy().crop(t0, t0), cov, sphere)[0]
    pos[ii] = dip.pos[0]

###############################################################################
# Compute localisation errors


actual_pos = 0.01 * np.array([[0.16, 1.61, 5.13],
                              [0.17, 1.35, 4.15],
                              [0.16, 1.05, 3.19],
                              [0.13, 0.80, 2.26]])
actual_pos = np.dot(actual_pos, [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

errors = 1e3 * np.linalg.norm(actual_pos - pos, axis=1)
print("errors (mm) : %s" % errors)

###############################################################################
# Plot the dipoles in 3D


def plot_pos(pos, color=(0., 0., 0.)):
    mlab.points3d(pos[:, 0], pos[:, 1], pos[:, 2], scale_factor=0.005,
                  color=color)


mne.viz.plot_alignment(evoked.info, bem=sphere, surfaces=[])
# Plot the position of the actual dipole
plot_pos(actual_pos, color=(1., 0., 0.))
# Plot the position of the estimated dipole
plot_pos(pos, color=(1., 1., 0.))
