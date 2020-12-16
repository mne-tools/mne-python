"""
==============================================
Generate a left cerebellum volume source space
==============================================

Generate a volume source space of the left cerebellum and plot its vertices
relative to the left cortical surface source space and the FreeSurfer
segmentation file.

"""

# Author: Alan Leggitt <alan.leggitt@ucsf.edu>
#
# License: BSD (3-clause)

import mne
from mne import setup_source_space, setup_volume_source_space
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
subject = 'sample'
aseg_fname = subjects_dir + '/sample/mri/aseg.mgz'

###############################################################################
# Setup the source spaces

# setup a cortical surface source space and extract left hemisphere
surf = setup_source_space(subject, subjects_dir=subjects_dir, add_dist=False)
lh_surf = surf[0]

# setup a volume source space of the left cerebellum cortex
volume_label = 'Left-Cerebellum-Cortex'
sphere = (0, 0, 0, 0.12)
lh_cereb = setup_volume_source_space(
    subject, mri=aseg_fname, sphere=sphere, volume_label=volume_label,
    subjects_dir=subjects_dir, sphere_units='m')

# Combine the source spaces
src = surf + lh_cereb

###############################################################################
# Plot the positions of each source space

fig = mne.viz.plot_alignment(subject=subject, subjects_dir=subjects_dir,
                             surfaces='white', coord_frame='head',
                             src=src)
mne.viz.set_3d_view(fig, azimuth=173.78, elevation=101.75,
                    distance=0.30, focalpoint=(-0.03, -0.01, 0.03))


###############################################################################
# You can export source positions to a NIfTI file::
#
#     >>> nii_fname = 'mne_sample_lh-cerebellum-cortex.nii'
#     >>> src.export_volume(nii_fname, mri_resolution=True)
#
# And display source positions in freeview::
#
#    >>> from mne.utils import run_subprocess
#    >>> mri_fname = subjects_dir + '/sample/mri/brain.mgz'
#    >>> run_subprocess(['freeview', '-v', mri_fname, '-v',
#                        '%s:colormap=lut:opacity=0.5' % aseg_fname, '-v',
#                        '%s:colormap=jet:colorscale=0,2' % nii_fname,
#                        '-slice', '157 75 105'])
