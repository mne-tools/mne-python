# -*- coding: utf-8 -*-
"""
.. _plot_source_alignment:

Source alignment and coordinate frames
======================================

This tutorial shows how to visually assess the spatial alignment of MEG sensor
locations, digitized scalp landmark and sensor locations, and MRI volumes. This
alignment process is crucial for computing the forward solution, as is
understanding the different coordinate frames involved in this process.

.. contents:: Page contents
   :local:
   :depth: 2

Let's start out by loading some data.
"""
import os.path as op

import numpy as np
import nibabel as nib
from scipy import linalg

import mne

data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
trans_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_raw-trans.fif')
raw = mne.io.read_raw_fif(raw_fname)
trans = mne.read_trans(trans_fname)
src = mne.read_source_spaces(op.join(subjects_dir, 'sample', 'bem',
                                     'sample-oct-6-src.fif'))

# load the T1 file and change the header information to the correct units
t1w = nib.load(op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz'))
t1w = nib.Nifti1Image(t1w.dataobj, t1w.affine)
t1w.header['xyzt_units'] = np.array(10, dtype='uint8')
t1_mgh = nib.MGHImage(t1w.dataobj, t1w.affine)

###############################################################################
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
#
# Understanding coordinate frames
# -------------------------------
# For M/EEG source imaging, there are three **coordinate frames** must be
# brought into alignment using two 3D `transformation matrices <wiki_xform_>`_
# that define how to rotate and translate points in one coordinate frame
# to their equivalent locations in another. The three main coordinate frames
# are:
#
# * :blue:`"meg"`: the coordinate frame for the physical locations of MEG
#   sensors
# * :gray:`"mri"`: the coordinate frame for MRI images, and scalp/skull/brain
#   surfaces derived from the MRI images
# * :pink:`"head"`: the coordinate frame for digitized sensor locations and
#   scalp landmarks ("fiducials")
#
#
# Each of these, plus a fourth coordinate frame "mri_voxel", are described in
# more detail below.
#
# A good way to start visualizing these coordinate frames is to use the
# `mne.viz.plot_alignment` function, which is used for creating or inspecting
# the transformations that bring these coordinate frames into alignment, and
# the resulting alignment of EEG sensors, MEG sensors, brain sources, and
# conductor models. If you provide ``subjects_dir`` and ``subject`` parameters,
# the function automatically loads the subject's Freesurfer MRI surfaces.
# Important for our purposes, passing ``show_axes=True`` to
# `~mne.viz.plot_alignment` will draw the origin of each coordinate frame in a
# different color, with axes indicated by different sized arrows:
#
# * shortest arrow: (**R**)ight / X
# * medium arrow: forward / (**A**)nterior / Y
# * longest arrow: up / (**S**)uperior / Z
#
# Note that all three coordinate systems are **RAS** coordinate frames and
# hence are also `right-handed`_ coordinate systems. Finally, note that the
# ``coord_frame`` parameter sets which coordinate frame the camera
# should initially be aligned with. Let's take a look:

fig = mne.viz.plot_alignment(raw.info, trans=trans, subject='sample',
                             subjects_dir=subjects_dir, surfaces='head-dense',
                             show_axes=True, dig=True, eeg=[], meg='sensors',
                             coord_frame='meg')
mne.viz.set_3d_view(fig, 45, 90, distance=0.6, focalpoint=(0., 0., 0.))
print('Distance from head origin to MEG origin: %0.1f mm'
      % (1000 * np.linalg.norm(raw.info['dev_head_t']['trans'][:3, 3])))
print('Distance from head origin to MRI origin: %0.1f mm'
      % (1000 * np.linalg.norm(trans['trans'][:3, 3])))
dists = mne.dig_mri_distances(raw.info, trans, 'sample',
                              subjects_dir=subjects_dir)
print('Distance from %s digitized points to head surface: %0.1f mm'
      % (len(dists), 1000 * np.mean(dists)))

###############################################################################
# Coordinate frame definitions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 1. Neuromag/Elekta/MEGIN head coordinate frame ("head", :pink:`pink axes`)
#      The head coordinate frame is defined through the coordinates of
#      anatomical landmarks on the subject's head: usually the Nasion (`NAS`_),
#      and the left and right preauricular points (`LPA`_ and `RPA`_).
#      Different MEG manufacturers may have different definitions of the head
#      coordinate frame. A good overview can be seen in the
#      `FieldTrip FAQ on coordinate systems`_.
#
#      For Neuromag/Elekta/MEGIN, the head coordinate frame is defined by the
#      intersection of
#
#      1. the line between the LPA (:red:`red sphere`) and RPA
#         (:purple:`purple sphere`), and
#      2. the line perpendicular to this LPA-RPA line one that goes through
#         the Nasion (:green:`green sphere`).
#
#      The axes are oriented as **X** origin→RPA, **Y** origin→NAS,
#      **Z** origin→upward (orthogonal to X and Y).
#
#      .. note:: The required 3D coordinates for defining the head coordinate
#                frame (NAS, LPA, RPA) are measured at a stage separate from
#                the MEG data recording. There exist numerous devices to
#                perform such measurements, usually called "digitizers". For
#                example, see the devices by the company `Polhemus`_.
#
# 2. MEG device coordinate frame ("meg", :blue:`blue axes`)
#      The MEG device coordinate frame is defined by the respective MEG
#      manufacturers. All MEG data is acquired with respect to this coordinate
#      frame. To account for the anatomy and position of the subject's head, we
#      use so-called head position indicator (HPI) coils. The HPI coils are
#      placed at known locations on the scalp of the subject and emit
#      high-frequency magnetic fields used to coregister the head coordinate
#      frame with the device coordinate frame.
#
#      From the Neuromag/Elekta/MEGIN user manual:
#
#          The origin of the device coordinate system is located at the center
#          of the posterior spherical section of the helmet with X axis going
#          from left to right and Y axis pointing front. The Z axis is, again
#          normal to the plane with positive direction up.
#
#      .. note:: The HPI coils are shown as :magenta:`magenta spheres`.
#                Coregistration happens at the beginning of the recording and
#                the head↔meg transformation matrix is stored in
#                ``raw.info['dev_head_t']``.
#
# 3. MRI coordinate frame ("mri", :gray:`gray axes`)
#      Defined by Freesurfer, the "MRI surface RAS" coordinate frame has its
#      origin at the center of a 256×256×256 1mm anisotropic volume (though the
#      center may not correspond to the anatomical center of the subject's
#      head).
#
#      .. note:: We typically align the MRI coordinate frame to the head
#                coordinate frame through a
#                `rotation and translation matrix <wiki_xform_>`_,
#                that we refer to in MNE as ``trans``.
#
# A bad example
# ^^^^^^^^^^^^^
# Let's try using `~mne.viz.plot_alignment` with ``trans=None``, which
# (incorrectly!) equates the MRI and head coordinate frames.

mne.viz.plot_alignment(raw.info, trans=None, subject='sample', src=src,
                       subjects_dir=subjects_dir, dig=True,
                       surfaces=['head-dense', 'white'], coord_frame='meg')

###############################################################################
# A good example
# ^^^^^^^^^^^^^^
# Here is the same plot, this time with the ``trans`` properly defined
# (using a precomputed transformation matrix).

mne.viz.plot_alignment(raw.info, trans=trans, subject='sample',
                       src=src, subjects_dir=subjects_dir, dig=True,
                       surfaces=['head-dense', 'white'], coord_frame='meg')

###############################################################################
# Visualizing the transformations
# -------------------------------
# Let's visualize these transformations using just the scalp surface and some
# digitized scalp points. To do this we'll write a custom function that takes
# a set of 3D points and plots them (along with the scalp surface) in the "mri"
# coordinate frame. Remember, *digitized scalp points* start out in the "head"
# coordinate frame, whereas *the scalp surface* comes from the MRI.


def plot_dig_alignment(points):
    points = points.copy()
    renderer = mne.viz.backends.renderer.create_3d_figure(
        size=(800, 400), bgcolor='w', scene=False)
    seghead_rr, seghead_tri = mne.read_surface(
        op.join(subjects_dir, 'sample', 'surf', 'lh.seghead'))
    renderer.mesh(*seghead_rr.T, triangles=seghead_tri, color=(0.7,) * 3,
                  opacity=0.2)
    for point in points:
        renderer.sphere(center=point, color='r', scale=5)
    mne.viz.set_3d_view(figure=renderer.figure, distance=1000,
                        focalpoint=(0., 0., 0.), elevation=90, azimuth=0)
    renderer.show()


###############################################################################
# Now that our function is defined, first we'll plot *untransformed* digitized
# scalp points (as if they were in the "mri" coordinate frame). You can see
# that the scalp points look good relative to each other, but the whole set of
# scalp points is mismatched to the MRI scalp surface. Note also that we're
# converting units before passing our points into the function: MNE-Python
# uses SI units internally (so, distances in meters) whereas the MRI space
# defined by Freesurfer is in millimeters.

head_space = np.array([dig['r'] for dig in raw.info['dig']], dtype=float)
plot_dig_alignment(head_space * 1e3)  # m → mm

###############################################################################
# Next, we'll apply the precomputed ``trans`` to convert the digitized points
# from the "head" to the "mri" coordinate frame. Since the MRI scalp surface
# is in RAS coordinates, the alignment fits as expected based on the
# coregistration. But, there's one more step, the RAS coordinates have to be
# transformed to match the mri aquisition which is in voxels.

mri_space = mne.transforms.apply_trans(trans, head_space, move=True)
plot_dig_alignment(mri_space * 1e3)  # m → mm

###############################################################################
# Finally, we'll apply a second transformation to the already-transformed
# digitized points. That transform comes from the T1 header, and converts from
# native MRI voxels (called the "mri_voxel" coordinate frame in MNE-Python) to
# the "mri" coordinate frame (MRI Surface RAS). Unlike MRI Surface RAS,
# "mri_voxel" has its origin in the corner of the volume (the left-most,
# posterior-most coordinate on the inferior-most MRI slice) instead of at the
# center of the volume. "mri_voxel" is also **not** an RAS coordinate system:
# rather, its XYZ directions are based on the acquisition order of the T1 image
# slices, so we'll need to do some extra steps here.
#
# .. note::
#     Normally, MNE-Python converts ``mri_voxel → mri`` coordinate frame
#     automatically before displaying skin/skull/brain surfaces, so the extra
#     steps aren't needed for most analysis tasks.

vox_to_mri = t1_mgh.header.get_vox2ras_tkr()
mri_to_vox = linalg.inv(vox_to_mri)

vox_space = mne.transforms.apply_trans(mri_to_vox, mri_space * 1e3)  # m → mm
vox_space = ((vox_space - 128) * [1, -1, 1])[:, [0, 2, 1]]
plot_dig_alignment(vox_space)


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
# to use a :ref:`spherical conductor model <eeg_sphere_model>`. It is
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
# .. _right-handed: https://en.wikipedia.org/wiki/Right-hand_rule
# .. _wiki_xform: https://en.wikipedia.org/wiki/Transformation_matrix
# .. _NAS: https://en.wikipedia.org/wiki/Nasion
# .. _LPA: http://www.fieldtriptoolbox.org/faq/how_are_the_lpa_and_rpa_points_defined/  # noqa:E501
# .. _RPA: http://www.fieldtriptoolbox.org/faq/how_are_the_lpa_and_rpa_points_defined/  # noqa:E501
# .. _Polhemus: https://polhemus.com/scanning-digitizing/digitizing-products/
# .. _FieldTrip FAQ on coordinate systems: http://www.fieldtriptoolbox.org/faq/how_are_the_different_head_and_mri_coordinate_systems_defined/  # noqa:E501
