# -*- coding: utf-8 -*-
# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#
# License: BSD (3-clause)
"""
Plot PSFs and CTFs with corresponding resolution metrics.

Compare metrics localisation error and spatial extent with the
corresponding PSFs and CTFs.
"""

import mne
from mne import SourceEstimate
from mne.datasets import sample
from mne.minimum_norm.resolution_matrix import (make_resolution_matrix,
                                                get_point_spread,
                                                get_cross_talk)
from mne.minimum_norm.resolution_metrics import (localisation_error_psf,
                                                 localisation_error_ctf)

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path + '/subjects/'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
fname_cov = data_path + '/MEG/sample/sample_audvis-cov.fif'
fname_evo = data_path + '/MEG/sample/sample_audvis-ave.fif'

# read forward solution
forward = mne.read_forward_solution(fname_fwd)
# forward operator with fixed source orientations
forward = mne.convert_forward_solution(forward, surf_ori=True,
                                       force_fixed=True)

# noise covariance matrix
noise_cov = mne.read_cov(fname_cov)

# evoked data for info
evoked = mne.read_evokeds(fname_evo, 0)

# make inverse operator from forward solution
# free source orientation
inverse_operator = mne.minimum_norm.make_inverse_operator(
    info=evoked.info, forward=forward, noise_cov=noise_cov, loose=0.,
    depth=None)

# regularisation parameter
snr = 3.0
lambda2 = 1.0 / snr ** 2
method = 'sLORETA'  # can be 'MNE', 'sLORETA' or 'eLORETA'

# compute resolution matrix for sLORETA
rm_lor = make_resolution_matrix(forward, inverse_operator,
                                method=method, lambda2=lambda2)

# compute resolution matrix for sLORETA
rm_lor = make_resolution_matrix(forward, inverse_operator,
                                method=method, lambda2=lambda2)

# Compute peak localisation error for PSFs and CTFs, respectively
locerr_psf = localisation_error_psf(rm_lor, inverse_operator['src'],
                                    metric='peak')
locerr_ctf = localisation_error_ctf(rm_lor, inverse_operator['src'],
                                    metric='peak')

# get PSF and CTF for sLORETA at one vertex
sources = [1000]

stc_psf = get_point_spread(rm_lor, forward['src'], sources, norm=True)

stc_ctf = get_cross_talk(rm_lor, forward['src'], sources, norm=True)

# Visualise individual PSF and CTF

# get vertices from source space
vertno_lh = forward['src'][0]['vertno']
vertno_rh = forward['src'][1]['vertno']
vertno = [vertno_lh, vertno_rh]

# Which vertex corresponds to selected source
verttrue = [vertno_lh[sources[0]]]  # just one vertex

# Find vertices with maxima in PSF and CTF
vert_max_psf = vertno_lh[stc_psf.data.argmax()]
vert_max_ctf = vertno_lh[stc_ctf.data.argmax()]

brain_psf = stc_psf.plot('sample', 'inflated', 'lh', subjects_dir=subjects_dir,
                         figure=1, title='PSF')

brain_psf.show_view('ventral')

# Indicate true source location for PSF
brain_psf.add_foci(verttrue, coords_as_verts=True, scale_factor=1., hemi='lh',
                   color='green')

# Indicate location of maximum of PSF
brain_psf.add_foci(vert_max_psf, coords_as_verts=True, scale_factor=1.,
                   hemi='lh', color='black')

brain_ctf = stc_ctf.plot('sample', 'inflated', 'lh', subjects_dir=subjects_dir,
                         figure=2)

brain_ctf.show_view('ventral')

# Indicate location of maximum of PSF
brain_ctf.add_foci(verttrue, coords_as_verts=True, scale_factor=1., hemi='lh',
                   color='green')

# Indicate location of maximum of PSF
brain_ctf.add_foci(vert_max_ctf, coords_as_verts=True, scale_factor=1.,
                   hemi='lh', color='black')

print('The green spheres indicate the true source location, and the black \
      spheres the maximum of the distribution.')
print('Peak localisation error for PSF: %.2f cm.' % (100. *
      locerr_psf[sources[0]]))
print('Peak localisation error for CTF: %.2f cm.' % (100. *
      locerr_ctf[sources[0]]))

# Visualise localisation error across the whole cortex for PSF and CTF

# First convert metric distributions to source estimate
# and convert localisation error from meter to centimetre
stc_le_psf = SourceEstimate(100. * locerr_psf, vertno, tmin=0., tstep=1.)
stc_le_ctf = SourceEstimate(100. * locerr_ctf, vertno, tmin=0., tstep=1.)

brain_le_psf = stc_le_psf.plot('sample', 'inflated', 'lh',
                               subjects_dir=subjects_dir, figure=3,
                               clim=dict(kind='value', lims=(0, 2, 4)),
                               title='PLE PSF')

brain_le_ctf = stc_le_ctf.plot('sample', 'inflated', 'lh',
                               subjects_dir=subjects_dir, figure=4,
                               clim=dict(kind='value', lims=(0, 2, 4)),
                               title='PLE CTF')
