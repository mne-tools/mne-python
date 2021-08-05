"""
=========================================
Use automated approach to coregistration
=========================================

This example shows how to use the coregistration functions to perform an
automated MEG-MRI coregistration.

.. warning:: The quality of the coregistration depends heavily upon the
             quality of the head shape collected during subject prepration and
             the quality of your T1-weighted MRI. Use with caution and check
             the coregistration error.
"""

# Author: Jon Houck <jon.houck@gmail.com>
#         Guillaume Favelier <guillaume.favelier@gmail.com>
#
# License: BSD-3-Clause

import os.path as op
import mne
from mne.coreg import Coregistration
from mne.io import read_info


data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
subject = 'sample'

fname_raw = op.join(data_path, 'MEG', subject, subject + '_audvis_raw.fif')
info = read_info(fname_raw)
plot_kwargs = dict(subject=subject, subjects_dir=subjects_dir,
                   surfaces=dict(head=0.4), dig=True, eeg=[],
                   meg=False, coord_frame='meg')
view_kwargs = dict(azimuth=45, elevation=90, distance=0.6,
                   focalpoint=(0., 0., 0.))

# %%
# Set up coreg model
coreg = Coregistration(info, subject, subjects_dir)
fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)
mne.viz.set_3d_view(fig, **view_kwargs)

# %%
# Do initial coreg fit for outlier detection
coreg.fit_fiducials(verbose=True)
fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)
mne.viz.set_3d_view(fig, **view_kwargs)

# %%
# Overweighting at this step also seems to throw off the fit for some datasets
coreg.fit_icp(n_iterations=6, nasion_weight=2., verbose=True)
fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)
mne.viz.set_3d_view(fig, **view_kwargs)

coreg.omit_hsp_points(distance=5. / 1000)  # distance is in meters

# %%
# Do final coreg fit
coreg.fit_icp(n_iterations=20, nasion_weight=10., verbose=True)
fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)
mne.viz.set_3d_view(fig, **view_kwargs)
