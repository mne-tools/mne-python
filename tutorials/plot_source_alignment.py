"""
.. _tut_source_alignment:

Source alignment
================

The aim of this tutorial is to show how to visually assess that the data are
well aligned in space for computing the forward solution.
"""
import os.path as op

import numpy as np
from mayavi import mlab

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
                       subjects_dir=subjects_dir, dig=True,
                       surfaces=['head-dense', 'brain'], coord_frame='meg')


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
#
# For more information, see step by step instructions
# `in these slides
# <http://www.slideshare.net/mne-python/mnepython-coregistration>`_.

# mne.gui.coregistration(subject='sample', subjects_dir=subjects_dir)
trans = mne.read_trans(tr_fname)
src = mne.read_source_spaces(op.join(data_path, 'MEG', 'sample',
                                     'sample_audvis-meg-oct-6-meg-inv.fif'))
mne.viz.plot_alignment(raw.info, trans=trans, subject='sample', src=src,
                       subjects_dir=subjects_dir, dig=True,
                       surfaces=['head-dense', 'white'], coord_frame='meg')

###############################################################################
# Visualizing coordinate frames
# -----------------------------
# If you are curious about the origin and orientation of each of these
# coordinate frames, you can use the ``show_axes`` argument to see how
# each coordinate frame is aligned (shortest arrow is right/X,
# medium is forward/anderior/Y, longest is up/superior/Z, i.e., RAS
# coordinates).
#
# From this plot, where we have removed surfaces to more clearly see the
# coordinate frame axes, you can see:
#
# 1. The Neuromage head coordinate frame that MNE uses (pink) is defined by
#    the intersection of the line between the LPA and RPA, and the normal from
#    that line to the Nasion.
# 2. The MEG device coordinate frame (cyan) is somewhat centered on the
#    sensors, and is aligned to the head coordinate frame during acquisition,
#    where it is stored in ``raw.info['dev_head_t']`` as a
#    :class:`mne.transforms.Transform`.
# 3. The MRI coordinate frame (gray), which is defined by Freesurfer during
#    reconstruction, is not the same as the head coordinate frame, and is
#    aligned to the head coordinate frame by ``trans``.

mne.viz.plot_alignment(raw.info, trans=trans, subject='sample',
                       subjects_dir=subjects_dir, surfaces='head-dense',
                       show_axes=True, dig=True, eeg=[], meg='sensors',
                       coord_frame='meg')
mlab.view(45, 90, distance=0.6, focalpoint=(0., 0., 0.))
print('Distance from head origin to MEG origin: %0.1f mm'
      % (1000 * np.linalg.norm(raw.info['dev_head_t']['trans'][:3, 3])))
print('Distance from head origin to MRI origin: %0.1f mm'
      % (1000 * np.linalg.norm(trans['trans'][:3, 3])))

###############################################################################
# Alignment without MRI
# ---------------------
# The surface alignments above are possible if you have the surfaces available
# from Freesurfer. :func:`mne.viz.plot_alignment` automatically searches for
# the correct surfaces from the provided ``subjects_dir``. Another option is
# to use a spherical conductor model. It is passed through ``bem`` parameter.

sphere = mne.make_sphere_model(info=raw.info, r0='auto', head_radius='auto')
src = mne.setup_volume_source_space(sphere=sphere, pos=10.)
mne.viz.plot_alignment(
    raw.info, eeg='projected', bem=sphere, src=src, dig=True,
    surfaces=['brain', 'outer_skin'], coord_frame='meg', show_axes=True)

###############################################################################
# It is also possible to use :func:`mne.gui.coregistration`
# to warp a subject (usually ``fsaverage``) to subject digitization data, see
# `these slides
# <http://www.slideshare.net/mne-python/mnepython-scale-mri>`_.
