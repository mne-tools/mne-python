"""
=====================
Plotting BEM Contours
=====================
This example reads in anatomical data
and BEM surfaces and plots the contours in different
colors. This is useful for inspecting the quality of
the BEM segmentations which are required for computing
the forward solution.
"""
# Author: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

from mne.viz import plot_bem_contours
from mne.datasets import sample

data_path = sample.data_path()

mri_fname = data_path + '/subjects/sample/mri/T1.mgz'

fname_is = data_path + '/subjects/sample/bem/sample-inner_skull-5120.surf'
fname_os = data_path + '/subjects/sample/bem/sample-outer_skull-5120.surf'
fname_s = data_path + '/subjects/sample/bem/sample-outer_skin-5120.surf'

surf_fnames = [fname_is, fname_os, fname_s]
slices = [200, 128, 80]

plot_bem_contours(mri_fname, surf_fnames, slices=slices)
