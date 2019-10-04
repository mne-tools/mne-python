"""
==============================================================
Compute spatial resolution metrics in source space.
==============================================================

Compute peak localisation error and spatial deviation for the point-spread
functions of dSPM and MNE.
"""
# Author: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#
# License: BSD (3-clause)

import mne
from mne.datasets import sample
from mne.minimum_norm.resolution_matrix import make_resolution_matrix
from mne.minimum_norm.resolution_metrics import resolution_metrics

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

# compute resolution matrix for MNE
rm_mne = make_resolution_matrix(forward, inverse_operator,
                                method='MNE', lambda2=lambda2)

# compute resolution matrix for sLORETA
rm_spm = make_resolution_matrix(forward, inverse_operator,
                                method='dSPM', lambda2=lambda2)

# Compute peak localisation error for PSFs
ple_mne_psf = resolution_metrics(rm_mne, inverse_operator['src'],
                                 function='psf',
                                 kind='localization_error', metric='peak')
ple_spm_psf = resolution_metrics(rm_spm, inverse_operator['src'],
                                 function='psf',
                                 kind='localization_error', metric='peak')

# Compute spatial deviation for PSFs
sd_mne_psf = resolution_metrics(rm_mne, inverse_operator['src'],
                                function='psf',
                                kind='spatial_extent', metric='sd')
sd_spm_psf = resolution_metrics(rm_spm, inverse_operator['src'],
                                function='psf',
                                kind='spatial_extent', metric='sd')

# Visualise peak localisation error (PLE) across the whole cortex for PSF

brain_le_mne = ple_mne_psf.plot('sample', 'inflated', 'lh',
                                subjects_dir=subjects_dir, figure=1,
                                clim=dict(kind='value', lims=(0, 2, 4)),
                                title='PLE MNE')

brain_le_spm = ple_spm_psf.plot('sample', 'inflated', 'lh',
                                subjects_dir=subjects_dir, figure=2,
                                clim=dict(kind='value', lims=(0, 2, 4)),
                                title='PLE dSPM')

# Subtract the two distributions and plot this difference
diff_ple = ple_mne_psf - ple_spm_psf

brain_le_diff = diff_ple.plot('sample', 'inflated', 'lh',
                              subjects_dir=subjects_dir, figure=3,
                              clim=dict(kind='value', pos_lims=(0., 1., 2.)),
                              title='PLE MNE-dSPM')

print('dSPM has generally lower peak localization error than MNE in deeper \
       brain areas (red color), but higher error (blue color) in more \
       superficial areas.')

# Visualise spatial deviation (SD) across the whole cortex for PSF

brain_le_mne = sd_mne_psf.plot('sample', 'inflated', 'lh',
                               subjects_dir=subjects_dir, figure=4,
                               clim=dict(kind='value', lims=(0, 2, 4)),
                               title='SD MNE')

brain_le_spm = sd_spm_psf.plot('sample', 'inflated', 'lh',
                               subjects_dir=subjects_dir, figure=5,
                               clim=dict(kind='value', lims=(0, 2, 4)),
                               title='SD dSPM')

# Subtract the two distributions and plot this difference
diff_sd = sd_mne_psf - sd_spm_psf

brain_sd_diff = diff_sd.plot('sample', 'inflated', 'lh',
                             subjects_dir=subjects_dir, figure=6,
                             clim=dict(kind='value', pos_lims=(0., 1., 2.)),
                             title='SD MNE-dSPM')

print('dSPM has generally higher spatial deviation than MNE (blue color), i.e. \
      worse performance to distinguish different sources.')
