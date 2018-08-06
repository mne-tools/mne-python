"""
=============================
Morph surface source estimate
=============================

This example demonstrates how to morph an individual subject
:class:`mne.SourceEstimate` to a common reference space. For this purpose we
will use :class:`mne.SourceMorph`. Pre-computed data will be morphed based on
an affine transformation on the surface vertices towards the reference
vertices. Here we will use 'fsaverage' as a reference space (see
`https://surfer.nmr.mgh.harvard.edu/fswiki/FsAverage`).

The transformation will be applied to the surface source estimate. The result
will be a plot showing the inflated surface representation of 'fsaverage',
overlaid with the morphed surface source estimate.

.. note:: For a tutorial about morphing see:
          :ref:`sphx_glr_auto_tutorials_plot_morph_stc.py`.
"""
# Author: Tommy Clausner <tommy.clausner@gmail.com>
#
# License: BSD (3-clause)
import os

from mne import read_source_estimate, SourceMorph
from mne.datasets import sample

print(__doc__)

###############################################################################
# Setup paths

sample_dir_raw = sample.data_path()
sample_dir = os.path.join(sample_dir_raw, 'MEG', 'sample')
subjects_dir = os.path.join(sample_dir_raw, 'subjects')

fname_stc = os.path.join(sample_dir, 'sample_audvis-meg')

###############################################################################
# Load example data

# Read stc from file
stc = read_source_estimate(fname_stc, subject='sample')

###############################################################################
# Morph SourceEstimate

# Initialize SourceMorph for SourceEstimate
morph = SourceMorph(subject_from='sample',
                    subject_to='fsaverage',
                    subjects_dir=subjects_dir)

# Morph data
stc_fsaverage = morph(stc)

###############################################################################
# Plot results

# Define plotting parameters
surfer_kwargs = dict(
    hemi='lh', subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
    initial_time=0.09, time_unit='s', size=(800, 800),
    smoothing_steps=5)

# Plot surface
brain = stc_fsaverage.plot(**surfer_kwargs)

# Add title
brain.add_text(0.1, 0.9, 'Morphed to fsaverage', 'title', font_size=16)
