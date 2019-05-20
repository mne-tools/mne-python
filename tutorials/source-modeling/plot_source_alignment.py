# -*- coding: utf-8 -*-
"""
.. _plot_source_alignment:

Source alignment and coordinate frames
======================================

The aim of this tutorial is to show how to visually assess that the data are
well aligned in space for computing the forward solution, and understand
the different coordinate frames involved in this process.

.. contents:: Topics
   :local:
   :depth: 2

Let's start out by loading some data.
"""
import os.path as op

import numpy as np
from mayavi import mlab

import mne
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
trans_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_raw-trans.fif')
raw = mne.io.read_raw_fif(raw_fname)
trans = mne.read_trans(trans_fname)
src = mne.read_source_spaces(op.join(subjects_dir, 'sample', 'bem',
                                     'sample-oct-6-src.fif'))

###############################################################################
# Understanding coordinate frames
# -------------------------------
# For M/EEG source imaging, there are three **coordinate frames** that we must
# bring into alignment using two 3D
# `transformation matrices <trans_matrices_>`_
# that define how to rotate and translate points in one coordinate frame
# to their equivalent locations in another.
#
# :func:`mne.viz.plot_alignment` is a very useful function for inspecting
# these transformations, and the resulting alignment of EEG sensors, MEG
# sensors, brain sources, and conductor models. If the ``subjects_dir`` and
# ``subject`` parameters are provided, the function automatically looks for the
# Freesurfer MRI surfaces to show from the subject's folder.
#
# We can use the ``show_axes`` argument to see the various coordinate frames
# given our transformation matrices. These are shown by axis arrows for each
# coordinate frame:
#
# * shortest arrow is (**R**)ight/X
# * medium is forward/(**A**)nterior/Y
# * longest is up/(**S**)uperior/Z
#
# i.e., a **RAS** coordinate system in each case. We can also set
# the ``coord_frame`` argument to choose which coordinate
# frame the camera should initially be aligned with.
#
# Let's take a look:

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
# Coordinate frame definitions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# .. raw:: html
#
#    <style>
#    .pink {color:DarkSalmon; font-weight:bold}
#    .blue {color:DeepSkyBlue; font-weight:bold}
#    .gray {color:Gray; font-weight:bold}
#    .magenta {color:Magenta; font-weight:bold}
#    .purple {color:Indigo; font-weight:bold}
#    .green {color:LimeGreen; font-weight:bold}
#    .red {color:Red; font-weight:bold}
#    </style>
#
# .. role:: pink
# .. role:: blue
# .. role:: gray
# .. role:: magenta
# .. role:: purple
# .. role:: green
# .. role:: red
#
# 1. Neuromag head coordinate frame ("head", :pink:`pink axes`)
#      Defined by the intersection of 1) the line between the LPA
#      (:red:`red sphere`) and RPA (:purple:`purple sphere`), and
#      2) the line perpendicular to this LPA-RPA line one that goes through
#      the Nasion (:green:`green sphere`).
#      The axes are oriented as **X** origin→RPA, **Y** origin→Nasion,
#      **Z** origin→upward (orthogonal to X and Y).
#
#      .. note:: This gets defined during the head digitization stage during
#                acquisition, often by use of a Polhemus or other digitizer.
#
# 2. MEG device coordinate frame ("meg", :blue:`blue axes`)
#      This is defined by the MEG manufacturers. From the Elekta user manual:
#
#          The origin of the device coordinate system is located at the center
#          of the posterior spherical section of the helmet with axis going
#          from left to right and axis pointing front. The axis is, again
#          normal to the plane with positive direction up.
#
#      .. note:: The device is coregistered with the head coordinate frame
#                during acquisition via emission of sinusoidal currents in
#                head position indicator (HPI) coils
#                (:magenta:`magenta spheres`) at the beginning of the
#                recording. This is stored in ``raw.info['dev_head_t']``.
#
# 3. MRI coordinate frame ("mri", :gray:`gray axes`)
#      Defined by Freesurfer, the MRI (surface RAS) origin is at the
#      center of a 256×256×256 1mm anisotropic volume (may not be in the center
#      of the head).
#
#      .. note:: This is aligned to the head coordinate frame that we
#                typically refer to in MNE as ``trans``.
#
# A bad example
# -------------
# Let's try using ``trans=None``, which (incorrectly!) equates the MRI
# and head coordinate frames.

mne.viz.plot_alignment(raw.info, trans=None, subject='sample', src=src,
                       subjects_dir=subjects_dir, dig=True,
                       surfaces=['head-dense', 'white'], coord_frame='meg')

###############################################################################
# It is quite clear that the MRI surfaces (head, brain) are not well aligned
# to the head digitization points (dots).
#
# A good example
# --------------
# Here is the same plot, this time with the ``trans`` properly defined
# (using a precomputed matrix).

mne.viz.plot_alignment(raw.info, trans=trans, subject='sample',
                       src=src, subjects_dir=subjects_dir, dig=True,
                       surfaces=['head-dense', 'white'], coord_frame='meg')

###############################################################################
# Defining the head↔MRI ``trans`` using the GUI
# ---------------------------------------------
# You can try creating the head↔MRI transform yourself using
# :func:`mne.gui.coregistration`.
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
# <https://www.slideshare.net/mne-python/mnepython-coregistration>`_.
# Uncomment the following line to align the data yourself.

# mne.gui.coregistration(subject='sample', subjects_dir=subjects_dir)

###############################################################################
# .. _plot_source_alignment_without_mri:
#
# Alignment without MRI
# ---------------------
# The surface alignments above are possible if you have the surfaces available
# from Freesurfer. :func:`mne.viz.plot_alignment` automatically searches for
# the correct surfaces from the provided ``subjects_dir``. Another option is
# to use a :ref:`spherical conductor model <ch_forward_spherical_model>`. It is
# passed through ``bem`` parameter.

sphere = mne.make_sphere_model(info=raw.info, r0='auto', head_radius='auto')
src = mne.setup_volume_source_space(sphere=sphere, pos=10.)
mne.viz.plot_alignment(
    raw.info, eeg='projected', bem=sphere, src=src, dig=True,
    surfaces=['brain', 'outer_skin'], coord_frame='meg', show_axes=True)

###############################################################################
# It is also possible to use :func:`mne.gui.coregistration`
# to warp a subject (usually ``fsaverage``) to subject digitization data, see
# `these slides
# <https://www.slideshare.net/mne-python/mnepython-scale-mri>`_.
#
# .. _trans_matrices: https://en.wikipedia.org/wiki/Transformation_matrix
