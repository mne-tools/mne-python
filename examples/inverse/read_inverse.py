"""
.. _ex-read-inverse:

===========================
Reading an inverse operator
===========================

The inverse operator's source space is shown in 3D.
"""
# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

import mne
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator
from mne.viz import set_3d_view

print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path / "subjects"
meg_path = data_path / "MEG" / "sample"
fname_trans = meg_path / "sample_audvis_raw-trans.fif"
inv_fname = meg_path / "sample_audvis-meg-oct-6-meg-inv.fif"

inv = read_inverse_operator(inv_fname)

print(f"Method: {inv['methods']}")
print(f"fMRI prior: {inv['fmri_prior']}")
print(f"Number of sources: {inv['nsource']}")
print(f"Number of channels: {inv['nchan']}")

src = inv["src"]  # get the source space

# Get access to the triangulation of the cortex

print(f"Number of vertices on the left hemisphere:  {len(src[0]['rr'])}")
print(f"Number of triangles on left hemisphere:     {len(src[0]['use_tris'])}")
print(f"Number of vertices on the right hemisphere: {len(src[1]['rr'])}")
print(f"Number of triangles on right hemisphere:    {len(src[1]['use_tris'])}")

# %%
# Show the 3D source space

fig = mne.viz.plot_alignment(
    subject="sample",
    subjects_dir=subjects_dir,
    trans=fname_trans,
    surfaces="white",
    src=src,
)
set_3d_view(fig, focalpoint=(0.0, 0.0, 0.06))
