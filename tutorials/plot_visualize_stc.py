"""
.. _tut_viz_epochs:

Visualize Source time courses
=============================

"""
# sphinx_gallery_thumbnail_number = 7

import os

import mne
from mne.datasets import sample

data_path = sample.data_path()
sample_dir = os.path.join(data_path, 'MEG', 'sample')
subjects_dir = os.path.join(data_path, 'subjects')

fname_stc = os.path.join(sample_dir, 'sample_audvis-meg')

###############################################################################
# Load example data

# Read stc from file
stc = mne.read_source_estimate(fname_stc, subject='sample')

###############################################################################
# This tutorial focuses on visualization of source time courses.
# All of the surface plots introduced here are based on PySurfer. PySurfer
# in turn uses the 3D visualization capabilities of Mayavi.
# As in the rest of ...
stc.plot()

###############################################################################
# blahblah
