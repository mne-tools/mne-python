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
import numpy as np
import mne
from mne.coreg import get_mni_fiducials, Coregistration
from mne.surface import dig_mri_distances
from mne.io import write_fiducials, read_info
from mne.io.constants import FIFF
from mne.transforms import write_trans

data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
subject = 'sample'
fname_raw = op.join(data_path, 'MEG', subject, subject + '_audvis_raw.fif')
fname_fids = op.join(subjects_dir, subject, 'bem', subject + '-fiducials.fif')
fname_trans = op.join(data_path, 'MEG', subject,
                      subject + 'audvis_raw_auto-trans.fif')
src = mne.read_source_spaces(op.join(subjects_dir, subject, 'bem',
                                     'sample-oct-6-src.fif'))

fids_mri = get_mni_fiducials(subject, subjects_dir)

# Save mri fiducials. This is mandatory. as fit_fiducials uses this file

write_fiducials(fname_fids, fids_mri, coord_frame=FIFF.FIFFV_COORD_MRI)
print('\nSaving estimated fiducials to %s \n' % fname_fids)

info = read_info(fname_raw)
# Set up coreg model
coreg = Coregistration(info, subject, subjects_dir)
# Do initial coreg fit for outlier detection
coreg.fit_fiducials()
# Overweighting at this step also seems to throw off the fit for some datasets
coreg.fit_icp(n_iterations=6, nasion_weight=2.)
coreg.omit_hsp_points(distance=5. / 1000)  # distance is in meters
# Do final coreg fit
coreg.fit_icp(n_iterations=20, nasion_weight=10.)
write_trans(fname_trans, coreg.trans)
errs_icp = coreg.compute_dig_head_distances()

raw = mne.io.Raw(fname_raw)
errs_nearest = dig_mri_distances(raw.info, fname_trans, subject, subjects_dir)

fig = mne.viz.plot_alignment(raw.info, trans=fname_trans, subject=subject,
                             subjects_dir=subjects_dir,
                             surfaces=dict(head=0.4),
                             dig=True, eeg=[], meg=False,
                             coord_frame='meg')
mne.viz.set_3d_view(fig, 45, 90, distance=0.6, focalpoint=(0., 0., 0.))

print('Median distance from digitized points to head surface is %.3f mm'
      % np.median(errs_icp * 1000))
print('''Median distance from digitized points to head surface using nearest
neighbor is %.3f mm''' % np.median(errs_nearest * 1000))
