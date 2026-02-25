"""
.. _tut-source-alignment:

======================================
Source alignment and coordinate frames
======================================

This tutorial shows how to visually assess the spatial alignment of MEG sensor
locations, MRI volumes, and digitized scalp landmark and sensor locations. This
alignment process is crucial for computing the forward solution, as is
understanding the different coordinate frames involved in this process.

Let's start out by loading some data.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

import nibabel as nib
import numpy as np

import mne

data_path = mne.datasets.sample.data_path()
subjects_dir = data_path / "subjects"
raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
trans_fname = data_path / "MEG" / "sample" / "sample_audvis_raw-trans.fif"
raw = mne.io.read_raw_fif(raw_fname)
trans = mne.read_trans(trans_fname)
src = mne.read_source_spaces(subjects_dir / "sample" / "bem" / "sample-oct-6-src.fif")

# Load the T1 file and change the header information to the correct units
t1w = nib.load(data_path / "subjects" / "sample" / "mri" / "T1.mgz")
t1w = nib.Nifti1Image(t1w.dataobj, t1w.affine)
t1w.header["xyzt_units"] = np.array(10, dtype="uint8")
t1_mgh = nib.MGHImage(t1w.dataobj, t1w.affine)

# %%
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
# Visualize elements requiring alignment and their coordinate frames
# ------------------------------------------------------------------
# Recall that a **coordinate frame** is a system for specifying locations using
# a framework described by an origin, a set of axes, and a scale of measurement.
# MEG data, MRI volumes, and digitized locations on a subject's head (such as
# fiducial points or EEG sensors) are all obtained using the coordinate frames
# of their respective acquisition devices, so they all start out in different
# frames. To perform source localization with M/EEG, these frames must be
# brought into alignment using two 3D `transformation matrices <wiki_xform_>`_
# that define how to rotate and translate points in one coordinate frame
# to their equivalent locations in another. The three main coordinate frames are:
#
# * :blue:`"meg"`: the coordinate frame for the physical locations of MEG sensors
# * :gray:`"mri"`: the coordinate frame for MRI images, as well as surfaces
#   derived from MRI images like the scalp, skull, and brain
# * :red:`"head"`: the coordinate frame for digitized sensor locations and
#   scalp landmarks ("fiducials")
#
#
# Elements that require alignment (for example, MEG sensors and MRI-derived
# surfaces) and their coordinate frames may be visualized with the
# `~mne.viz.plot_alignment` function. Passing ``show_axes=True`` to
# `~mne.viz.plot_alignment` will draw coordinate frame axes of the
# appropriate color for the frame types above, using arrow length
# to differentiate the positive direction for each axis as follows:
#
# * shortest arrow: (**R**)ight / X axis
# * medium arrow: forward / (**A**)nterior / Y axis
# * longest arrow: up / (**S**)uperior / Z axis
#
# Note that all three coordinate systems are **RAS** coordinate frames and
# hence are also right-handed coordinate systems (i.e. positive rotation
# around the z axis is counter-clockwise when viewing the x-y plane from a
# positive location on the z axis). When plotting, the viewer camera aligns
# with the coordinate frame passed to the 'coord_frame' parameter such that the
# origin of the given frame is centered and the y-axis points directly at the
# camera. Let's have a look, aligning the camera to the MEG coordinate frame:

fig = mne.viz.plot_alignment(
    raw.info,
    trans=trans,
    subject="sample",
    subjects_dir=subjects_dir,
    surfaces="head-dense",
    show_axes=True,
    dig=True,
    eeg=[],
    meg="sensors",
    coord_frame="meg",
    mri_fiducials="estimated",
)

# %%
# For comparison, let's see what happens when we align the camera view with
# head coordinates:

fig = mne.viz.plot_alignment(
    raw.info,
    trans=trans,
    subject="sample",
    subjects_dir=subjects_dir,
    surfaces="head-dense",
    show_axes=True,
    dig=True,
    eeg=[],
    meg="sensors",
    coord_frame="head",
    mri_fiducials="estimated",
)

# %%
# Aligning the camera view to the head coordinate system makes the MEG sensors
# appear tilted as the sample subject's head leaned right during acquisition.
#
# The camera view can be manually set to optimize visibility of specific
# features required to check alignment. A side view of the face makes it easy
# to check that the head position inside the MEG helmet is appropriate.

fig = mne.viz.plot_alignment(
    raw.info,
    trans=trans,
    subject="sample",
    subjects_dir=subjects_dir,
    surfaces="head-dense",
    show_axes=True,
    dig=True,
    eeg=[],
    meg="sensors",
    coord_frame="head",
    mri_fiducials="estimated",
)

mne.viz.set_3d_view(fig, 45, 90, distance=0.6, focalpoint=(0.0, 0.0, 0.0))
print(
    "Distance from head origin to MEG origin: "
    f"{1000 * np.linalg.norm(raw.info['dev_head_t']['trans'][:3, 3]):.1f} mm"
)
print(
    "Distance from head origin to MRI origin: "
    f"{1000 * np.linalg.norm(trans['trans'][:3, 3]):.1f} mm"
)
dists = mne.dig_mri_distances(raw.info, trans, "sample", subjects_dir=subjects_dir)
print(
    f"Distance from {len(dists)} digitized points to head surface: "
    f"{1000 * np.mean(dists):0.1f} mm"
)

# Screenshots of the 3D alignment plots from various camera perspectives
# can be saved using the figure plotter's 'save_graphic' method, e.g.
# `fig.plotter.save_graphic(save_path)`.


# %%
# Visually assess alignment quality
# ---------------------------------
#
# Plotting alignment makes it easy to identify various alignment problems.
#
# **Alignment problem #1: Bad MEG -> head transform**
# If digitized points map correctly to the head surface, but both head and dig
# points are misaligned to the MEG sensors, there may be a problem with the
# transform relating the MEG sensor locations to the head coordinate frame.
# This can happen, for example, when cHPI coils are mis-localized. We
# can visualize bad MEG -> head transforms by corrupting the raw data's correct
# transform with a 90 degree rotation around the x-axis and plotting the result.

rot_matrix = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

# construct an info instance with a corrupted dev_head_t
bad_info = raw.info.copy()
orig_dev_head_t = raw.info["dev_head_t"]["trans"]
bad_dev_head_t = orig_dev_head_t @ rot_matrix
bad_info["dev_head_t"]["trans"] = bad_dev_head_t

# visualize results
fig = mne.viz.plot_alignment(
    bad_info,
    trans=trans,
    subject="sample",
    src=None,
    subjects_dir=subjects_dir,
    dig=True,
    surfaces=[
        "head-dense",
    ],
    coord_frame="meg",
    show_axes=True,
)
mne.viz.set_3d_view(fig, -180, 90, distance=0.8, focalpoint=(0.0, 0.0, 0.0))

# %%
# **Alignment problem #2: Bad MRI -> head transform**
# If digitized points float off the surface of the head, or fiducial points are
# misplaced, this suggests a bad coregistration.

# construct a corrupt trans
bad_trans = trans.copy()
bad_trans["trans"] = trans["trans"] @ rot_matrix

# visualize results
fig = mne.viz.plot_alignment(
    raw.info,
    trans=bad_trans,
    subject="sample",
    src=src,
    subjects_dir=subjects_dir,
    dig=True,
    surfaces=["head-dense", "white"],
    coord_frame="meg",
)
mne.viz.set_3d_view(fig, -180, 90, distance=0.8, focalpoint=(0.0, 0.0, 0.0))

# %%
# Note that, while both types of alignment errors make the head look misaligned
# to the MEG sensors, they can be differentiated by whether or not digitized
# points sit properly on the head surface.
#
# With the skills from this tutorial,
# you can plot elements that require alignment, set the view to perform the
# checks you need, and finally assess the quality of the fits you see. For a
# more detailed explanation of using MRI-generated surfaces in MNE, see the
# :ref:`_tut-freesurfer-reconstruction` tutorial.

# .. _wiki_xform: https://en.wikipedia.org/wiki/Transformation_matrix
