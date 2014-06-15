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

from mne.viz import plot_bem
from mne.datasets import sample

data_path = sample.data_path()

plot_bem(subject='sample', subjects_dir=data_path, orientation='coronal')
