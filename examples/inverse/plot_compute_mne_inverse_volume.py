"""
=======================================================================
Compute MNE-dSPM inverse solution on evoked data in volume source space
=======================================================================

Compute dSPM inverse solution on MNE evoked dataset in a volume source
space and stores the solution in a nifti file for visualisation.

"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt

from nilearn.plotting import plot_stat_map
from nilearn.image import index_img

from mne.datasets import sample
from mne import read_evokeds
from mne.minimum_norm import apply_inverse, read_inverse_operator

print(__doc__)

data_path = sample.data_path()
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-vol-7-meg-inv.fif'
fname_evoked = data_path + '/MEG/sample/sample_audvis-ave.fif'
subjects_dir = data_path + '/subjects'
fname_t1 = subjects_dir + '/sample/mri/T1.mgz'

snr = 3.0
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

# Load data
evoked = read_evokeds(fname_evoked, condition=0, baseline=(None, 0))
inverse_operator = read_inverse_operator(fname_inv)
src = inverse_operator['src']

# Compute inverse solution
stc = apply_inverse(evoked, inverse_operator, lambda2, method)
stc.crop(0.0, 0.2)

# Export result as a 4D NIfTI object
img = stc.as_volume(src,
                    mri_resolution=False)  # set True for full MRI resolution

# Save it as a NIfTI file:
# nib.save(img, 'mne_%s_inverse.nii.gz' % method)

###############################################################################
# Plot with our wrapper to :func:`nilearn.plotting.plot_glass_brain`:

stc.plot_glass_brain(subject='sample', subjects_dir=subjects_dir,
                     src=src, vmin=8, vmax=26)

###############################################################################
# Plot directly with :func:`nilearn.plotting.plot_stat_map`:

plot_stat_map(index_img(img, 61), fname_t1, threshold=8.,
              title='%s (t=%.1f s.)' % (method, stc.times[61]))
plt.show()
