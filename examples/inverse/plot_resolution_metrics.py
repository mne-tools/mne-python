# -*- coding: utf-8 -*-
# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)
"""
Plot PSFs and CTFs with corresponding resolution metrics.

Compare metrics localisation error and spatial extent with the
corresponding PSFs and CTFs.
"""

# WIP

import mne
from mne.datasets import sample
from mne.minimum_norm.resolution_matrix import (make_resolution_matrix,
                                                get_psf_ctf_vertex)
from mne.minimum_norm.resolution_metrics import localisation_error

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
method = 'MNE'  # can be 'MNE' or 'sLORETA'

# compute resolution matrix for sLORETA
rm_lor = make_resolution_matrix(forward, inverse_operator,
                                method='sLORETA', lambda2=lambda2)

# compute resolution matrix for sLORETA
rm_lor = make_resolution_matrix(forward, inverse_operator,
                                method='sLORETA', lambda2=lambda2)

locerr_psf = localisation_error(rm_lor, inverse_operator['src'], type='psf',
                            metric='peak')
locerr_ctf = localisation_error(rm_lor, inverse_operator['src'], type='ctf',
                            metric='peak')

# get PSF and CTF for sLORETA at one vertex
sources = [1000]

stc_psf = get_psf_ctf_vertex(rm_lor, forward['src'], sources, 'psf',
                             norm=True)

stc_ctf = get_psf_ctf_vertex(rm_lor, forward['src'], sources, 'ctf',
                             norm=True)

# Visualise
from mayavi import mlab

txt = 'PLE: %.2f cm' # localisation error for display

# Which vertex corresponds to selected source
vertno_lh = forward['src'][0]['vertno']
verttrue = [vertno_lh[sources[0]]]  # just one vertex

# # maximum value for scaling
# mp = np.abs(stc_psf.data).max()
# mc = np.abs(stc_ctf.data).max()

# find vertices with maxima in PSF and CTF
vert_max_psf = vertno_lh[stc_psf.data.argmax()]
vert_max_ctf = vertno_lh[stc_ctf.data.argmax()]

brain_psf = stc_psf.plot('sample', 'inflated', 'lh', subjects_dir=subjects_dir,
                         figure=1, title='PSF')
mlab.title('sLORETA PSF')
mlab.text(.25, .15, txt % (100.*locerr_psf[sources[0]]))

brain_psf.show_view('ventral')

# True source location for PSF
brain_psf.add_foci(verttrue, coords_as_verts=True, scale_factor=1., hemi='lh',
                   color='green')

# Maximum of PSF
brain_psf.add_foci(vert_max_psf, coords_as_verts=True, scale_factor=1.,
                   hemi='lh', color='black')

brain_ctf = stc_ctf.plot('sample', 'inflated', 'lh', subjects_dir=subjects_dir,
                         figure=2)
mlab.title('sLORETA CTF')
mlab.text(.25, .15, txt % (100.*locerr_ctf[sources[0]]))

brain_ctf.show_view('ventral')

brain_ctf.add_foci(verttrue, coords_as_verts=True, scale_factor=1., hemi='lh',
                   color='green')

# Maximum of CTF
brain_ctf.add_foci(vert_max_ctf, coords_as_verts=True, scale_factor=1.,
                   hemi='lh', color='black')

print('The green spheres indicate the true source location, and the black \
      spheres the maximum of the distribution.')
