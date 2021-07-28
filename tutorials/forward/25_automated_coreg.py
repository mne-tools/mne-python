"""
=========================================
Use automated approach to co-registration
=========================================

This example shows how to use the get_mni_fiducials routine and the
coregistration functions to perform an automated MEG-MRI co-registration.

.. warning:: The quality of the co-registration depends heavily upon the
             quality of the head shape collected during subject prepration and
             the quality of your T1-weighted MRI. Use with caution and check
             the co-registration error.
"""

# Author: Jon Houck <jon.houck@gmail.com>
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

# Set up coreg model
coreg = Coregistration(info, subject, subjects_dir)
# Do initial coreg fit for outlier detection
coreg.fit_fiducials(verbose=True)
kwargs = dict(azimuth=45, elevation=90, distance=0.6, focalpoint=(0., 0., 0.))
fig = mne.viz.plot_alignment(info, trans=coreg.trans, subject=subject,
                             subjects_dir=subjects_dir,
                             surfaces=dict(head=0.4),
                             dig=True, eeg=[], meg=False,
                             coord_frame='meg')
mne.viz.set_3d_view(fig, **kwargs)

# Overweighting at this step also seems to throw off the fit for some datasets
coreg.fit_icp(n_iterations=6, nasion_weight=2., verbose=True)
fig = mne.viz.plot_alignment(info, trans=coreg.trans, subject=subject,
                             subjects_dir=subjects_dir,
                             surfaces=dict(head=0.4),
                             dig=True, eeg=[], meg=False,
                             coord_frame='meg')
mne.viz.set_3d_view(fig, **kwargs)

coreg.omit_hsp_points(distance=5. / 1000)  # distance is in meters
# Do final coreg fit
coreg.fit_icp(n_iterations=20, nasion_weight=10., verbose=True)
fig = mne.viz.plot_alignment(info, trans=coreg.trans, subject=subject,
                             subjects_dir=subjects_dir,
                             surfaces=dict(head=0.4),
                             dig=True, eeg=[], meg=False,
                             coord_frame='meg')
mne.viz.set_3d_view(fig, **kwargs)
