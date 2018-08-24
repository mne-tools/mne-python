"""
=============================
Morph surface source estimate
=============================

This example demonstrates how to morph an individual subject's
:class:`mne.SourceEstimate` to a common reference space. We achieve this using
:class:`mne.SourceMorph`. Pre-computed data will be morphed based on
a spherical representation of the cortex computed using the spherical
registration of
:ref:`FreeSurfer <sphx_glr_auto_tutorials_plot_background_freesurfer.py>`
(https://surfer.nmr.mgh.harvard.edu/fswiki/SurfaceRegAndTemplates). This
transform will be used to morph the surface vertices of the subject towards the
reference vertices. Here we will use 'fsaverage' as a reference space (see
https://surfer.nmr.mgh.harvard.edu/fswiki/FsAverage).

The transformation will be applied to the surface source estimate. A plot
depicting the successful morph will be created for the spherical and inflated
surface representation of ``'fsaverage'``, overlaid with the morphed surface
source estimate.

.. note:: For a tutorial about morphing see:
          :ref:`sphx_glr_auto_tutorials_plot_morph_stc.py`.
"""
# Author: Tommy Clausner <tommy.clausner@gmail.com>
#
# License: BSD (3-clause)
import os

import mne
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
stc = mne.read_source_estimate(fname_stc, subject='sample')

###############################################################################
# Setting up SourceMorph for SourceEstimate
# -----------------------------------------
#
# In MNE surface source estimates are represented as lists of vertices. If
# that is not entirely clear, we ask ourselves
# :ref:`sphx_glr_auto_tutorials_plot_object_source_estimate.py`
#
# The respective list of our data can either be obtained from
# :class:`mne.SourceSpaces` (src) or from the data we want to morph itself. If
# src is not provided, the morph will not be precomputed but instead will be
# prepared for morphing when calling.
#
# This works only with (Vector) :class:`SourceEstimate <mne.SourceEstimate>`.
# See :class:`mne.SourceMorph` for additional parameter settings. We keep the
# default parameters for *src* and *spacing*.
#
# Since the default of spacing (resolution of surface mesh) is 5 and subject_to
# was set to 'fsaverage', SourceMorph will use default vertices to morph
# (``[np.arange(10242)] * 2``).
#
# If src was not defined, the morph will actually not be precomputed, because
# we lack the vertices *from* that we want to compute. Instead the morph will
# be set up and when applying it, the actual transformation will be computed on
# the fly.
#
# Initialize SourceMorph for SourceEstimate

morph = mne.compute_source_morph(subject_from='sample', subject_to='fsaverage',
                                 src=stc, subjects_dir=subjects_dir)

###############################################################################
# Apply morph to (Vector) SourceEstimate
# --------------------------------------
#
# The morph will be applied to the source estimate data, by giving it as the
# first argument to the morph we computed above.

stc_fsaverage = morph(stc)

###############################################################################
# Plot results
# ------------

# Define plotting parameters
surfer_kwargs = dict(
    hemi='lh', subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
    initial_time=0.09, time_unit='s', size=(800, 800),
    smoothing_steps=5)

# As spherical surface
brain = stc_fsaverage.plot(surface='sphere', **surfer_kwargs)

# Add title
brain.add_text(0.1, 0.9, 'Morphed to fsaverage (spherical)', 'title',
               font_size=16)

###############################################################################
# As inflated surface
brain_inf = stc_fsaverage.plot(surface='inflated', **surfer_kwargs)

# Add title
brain_inf.add_text(0.1, 0.9, 'Morphed to fsaverage (inflated)', 'title',
                   font_size=16)

###############################################################################
# Reading and writing SourceMorph from and to disk
# ------------------------------------------------
#
# An instance of SourceMorph can be saved, by calling
# :meth:`morph.save <mne.SourceMorph.save>`.
#
# This methods allows for specification of a filename under which the ``morph``
# will be save in ".h5" format. If no file extension is provided, "-morph.h5"
# will be appended to the respective defined filename::
#
#     >>> morph.save('my-file-name')
#
# Reading a saved source morph can be achieved by using
# :func:`mne.read_source_morph`::
#
#     >>> morph = mne.read_source_morph('my-file-name-morph.h5')
#
# Once the environment is set up correctly, no information such as
# ``subject_from`` or ``subjects_dir`` must be provided, since it can be
# inferred from the data and use morph to 'fsaverage' by default. SourceMorph
# can further be used without creating an instance and assigning it to a
# variable. Instead the :class:`__init__ <mne.SourceMorph>` and
# :meth:`__call__ <mne.SourceMorph.__call__>` methods of SourceMorph can be
# easily chained into a handy one-liner. Taking this together the shortest
# possible way to morph data directly would be:

stc_fsaverage = mne.compute_source_morph(subjects_dir=subjects_dir,
                                         src=stc)(stc)
