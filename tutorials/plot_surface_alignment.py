"""
.. _tut_surface_alignment:

Surface alignment
=================

The aim of this tutorial is to show how to visually assess that the surfaces
are well aligned in space.
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
# the surface alignment before source analysis. The function automatically
# looks for the Freesurfer surfaces from the subject's folder.
src = mne.read_source_spaces(op.join(data_path, 'MEG', 'sample',
                                     'sample_audvis-meg-oct-6-meg-inv.fif'))
mne.viz.plot_alignment(raw.info, trans=None, subject='sample',
                       subjects_dir=subjects_dir, src=src)


###############################################################################
# It is quite clear that things are not well aligned for estimating the
# sources. We need to provide the function with a transformation that aligns
# the MRI with the MEG data. Here we use a precomputed matrix, but you can try
# create it yourself using :func:`mne.gui.coregistration`.
trans = mne.read_trans(tr_fname)

###############################################################################
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
# * Click ``Fit Head Shape``. This will align the digitization points to the
#   head surface. Sometimes the fitting algorithm doesn't find the correct
#   alignment immediately. You can try first fitting using LPA/RPA or fiducials
#   and then align according to the digitization. You can also finetune
#   manually with the controls on the right side of the panel.
# * Click ``Save As...`` (lower right corner of the panel), set the filename
#   and read it with :func:`mne.read_trans`.

# mne.gui.coregistration(subject='sample', subjects_dir=subjects_dir)
mne.viz.plot_alignment(raw.info, trans=trans, subject='sample',
                       subjects_dir=subjects_dir, src=src)
