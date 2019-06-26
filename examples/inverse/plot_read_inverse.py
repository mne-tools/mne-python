"""
===========================
Reading an inverse operator
===========================

The inverse operator's source space is shown in 3D.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import mne
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'

fname_trans = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'

inv_fname = data_path
inv_fname += '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'

inv = read_inverse_operator(inv_fname)

print("Method: %s" % inv['methods'])
print("fMRI prior: %s" % inv['fmri_prior'])
print("Number of sources: %s" % inv['nsource'])
print("Number of channels: %s" % inv['nchan'])

src = inv['src']  # get the source space

# Get access to the triangulation of the cortex

print("Number of vertices on the left hemisphere: %d" % len(src[0]['rr']))
print("Number of triangles on left hemisphere: %d" % len(src[0]['use_tris']))
print("Number of vertices on the right hemisphere: %d" % len(src[1]['rr']))
print("Number of triangles on right hemisphere: %d" % len(src[1]['use_tris']))

###############################################################################
# Show result on 3D source space

mne.viz.plot_alignment(subject='sample', subjects_dir=subjects_dir,
                       trans=fname_trans, surfaces='white', src=src)
