# Author: Alan Leggitt <alan.leggitt@ucsf.edu>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nibabel as nib
from mne import (setup_source_space, setup_volume_source_space)
from mne.datasets import sample
from mne.utils import run_subprocess

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
subj = 'sample'
aseg_fname = subjects_dir + '/sample/mri/aseg.mgz'
mri_fname = subjects_dir + '/sample/mri/brain.mgz'

# setup a cortical surface source space
surf = setup_source_space(subj, subjects_dir=subjects_dir, add_dist=False,
                          overwrite=True)

# setup a volume source space of the left cortical white matter
label = 'Left-Cerebral-White-Matter'
lh_ctx = setup_volume_source_space(subj, mri=aseg_fname, volume_label=label,
                                   subjects_dir=subjects_dir)

#########################################
# Plot the positions of each source space

# setup a 3d axis
ax = plt.axes(projection='3d')

# plot the surface sources
x1, y1, z1 = surf[0]['rr'].T
ax.plot(x1, y1, z1, 'bo', alpha=0.1)

# plot the white matter sources
x2, y2, z2 = lh_ctx[0]['rr'][lh_ctx[0]['inuse'].astype(bool)].T
ax.plot(x2, y2, z2, 'ro', alpha=0.5)

plt.show()

###################################
# Export the volume source to nifti

# tranform vertices to 3d volume
shape = lh_ctx[0]['shape']
shape3d = (shape[2], shape[1], shape[0])
vol = lh_ctx[0]['inuse'].reshape(shape3d)
# calculate affine transform (similar to source_estimate.save_stc_as_volume)
aff = lh_ctx[0]['src_mri_t']['trans']
aff = np.dot(lh_ctx[0]['mri_ras_t']['trans'], aff)
aff[:3] *= 1e3  # convert to millimeters

# setup the nifti file
hdr = nib.Nifti1Header()
hdr.set_xyzt_units('mm')
img = nib.Nifti1Image(vol, affine=aff, header=hdr)

# save as nifti file
nii_fname = 'mne_sample_lh-cortical-white-matter.nii'
nib.save(img, nii_fname)

# display image in freeview
run_subprocess(['freeview', '-v', mri_fname, '-v',
                '%s:colormap=lut:opacity=0.5' % aseg_fname, '-v',
                '%s:colormap=jet:colorscale=0,2' % nii_fname, '-slice',
                '157 75 105'])
