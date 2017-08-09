"""
.. _tut_source_alignment:

Source alignment
================

The aim of this tutorial is to show how to visually assess that the data
are well aligned in space for computing the forward solution.
"""
import os.path as op

import mne
from mne.datasets import sample

print(__doc__)


###############################################################################
# Set parameters
# --------------
data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
tr_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-trans.fif')
raw = mne.io.read_raw_fif(raw_fname)


###############################################################################
# :func:`mne.viz.plot_alignment` is a very useful function for inspecting
# the surface alignment before source analysis. If the ``subjects_dir`` and
# ``subject`` parameters are provided, the function automatically looks for the
# Freesurfer surfaces from the subject's folder. Here we use trans=None, which
# (incorrectly!) equates the MRI and head coordinate frames.
mne.viz.plot_alignment(raw.info, trans=None, subject='sample',
                       subjects_dir=subjects_dir, surfaces=['head', 'brain'])


###############################################################################
# It is quite clear that things are not well aligned for estimating the
# sources. We need to provide the function with a transformation that aligns
# the MRI with the MEG data. Here we use a precomputed matrix, but you can try
# creating it yourself using :func:`mne.gui.coregistration`.
#
# Aligning the data using GUI
# ---------------------------
# Uncomment the following line to align the data yourself.
#
# * First you must load the digitization data from the raw file
#   (``Head Shape Source``). The MRI data is already loaded if you provide the
#   ``subject`` and ``subjects_dir``. Toggle ``Always Show Head Points`` to see
#   the digitization points.
# * To set the landmarks, toggle ``Edit`` radio button in ``MRI Fiducials``.
# * Set the landmarks by clicking the radio button (LPA, Nasion, RPA) and then
#   clicking the corresponding point in the image.
# * After doing this for all the landmarks, toggle ``Lock`` radio button. You
#   can omit outlier points, so that they don't interfere with the finetuning.
#
#   .. note:: You can save the fiducials to a file and pass
#             ``mri_fiducials=True`` to plot them in
#             :func:`mne.viz.plot_alignment`. The fiducials are saved to the
#             subject's bem folder by default.
# * Click ``Fit Head Shape``. This will align the digitization points to the
#   head surface. Sometimes the fitting algorithm doesn't find the correct
#   alignment immediately. You can try first fitting using LPA/RPA or fiducials
#   and then align according to the digitization. You can also finetune
#   manually with the controls on the right side of the panel.
# * Click ``Save As...`` (lower right corner of the panel), set the filename
#   and read it with :func:`mne.read_trans`.

# mne.gui.coregistration(subject='sample', subjects_dir=subjects_dir)
trans = mne.read_trans(tr_fname)
src = mne.read_source_spaces(op.join(data_path, 'MEG', 'sample',
                                     'sample_audvis-meg-oct-6-meg-inv.fif'))
mne.viz.plot_alignment(raw.info, trans=trans, subject='sample', src=src,
                       subjects_dir=subjects_dir, surfaces=['head', 'white'])


###############################################################################
# The previous is possible if you have the surfaces available from Freesurfer.
# The function automatically searches for the correct surfaces from the
# provided ``subjects_dir``. Otherwise it is possible to use the sphere
# conductor model. It is passed through ``bem`` parameter.
#
# .. note:: ``bem`` also accepts bem solutions (:func:`mne.read_bem_solution`)
#           or a list of bem surfaces (:func:`mne.read_bem_surfaces`).
sphere = mne.make_sphere_model(info=raw.info, r0='auto', head_radius='auto')
mne.viz.plot_alignment(raw.info, subject='sample', eeg='projected',
                       meg='helmet', bem=sphere, dig=True,
                       surfaces=['brain', 'inner_skull', 'outer_skull',
                                 'outer_skin'])


###############################################################################
# For more information see step by step instructions
# `for subjects with structural MRI
# <http://www.slideshare.net/mne-python/mnepython-coregistration>`_ and `for
# subjects for which no MRI is available
# <http://www.slideshare.net/mne-python/mnepython-scale-mri>`_.
