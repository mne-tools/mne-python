"""
=====================
Plotting BEM Contours
=====================

This example displays the BEM surfaces (inner skull, outer skull,
outer skin) as yellow contours on top of the T1 MRI anatomical image
used for segmentation. This is useful for inspecting the quality of the
BEM segmentations which are required for computing the forward solution.
"""

# Author: Mainak Jas <mainak@neuro.hut.fi>
#         Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

from mne.viz import plot_bem
from mne.datasets import sample

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'

plot_bem(subject='sample', subjects_dir=subjects_dir, orientation='axial')
plot_bem(subject='sample', subjects_dir=subjects_dir, orientation='sagittal')
plot_bem(subject='sample', subjects_dir=subjects_dir, orientation='coronal')
