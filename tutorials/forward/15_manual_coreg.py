"""
.. _tut-manual-coreg:

============================================
Manually Co-register Digitized Points to MRI
============================================

This tutorial shows how to perform a manual coregistration and save the results
in a 'trans.fif' file for use in forward model creation. We examine two
common scenarios: first, coregistration with an individual MRI, and second,
coregistration for subjects using a template source space instead of an MRI.

Let's start out by loading some data.
"""

import mne

data_path = mne.datasets.sample.data_path()
subjects_dir = data_path / "subjects"
raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
trans_fname = data_path / "MEG" / "sample" / "sample_audvis_raw-trans.fif"
raw = mne.io.read_raw_fif(raw_fname)

# .. _creating-trans:
#
# Defining the head↔MRI ``trans`` using the GUI
# ---------------------------------------------
# You can try creating the head↔MRI transform yourself using
# :func:`mne.gui.coregistration`.
#
# * To set the MRI fiducials, make sure ``Lock Fiducials`` is toggled off.
# * Set the landmarks by clicking the radio button (LPA, Nasion, RPA) and then
#   clicking the corresponding point in the image.
#
# .. note::
#    The position of each fiducial used is the center of the octahedron icon.
#
# * After doing this for all the landmarks, toggle ``Lock Fiducials`` radio
#   button and optionally pressing ``Save MRI Fid.`` which will save to a
#   default location in the ``bem`` folder of the Freesurfer subject directory.
# * Then you can load the digitization data from the raw file
#   (``Path to info``).
# * Click ``Fit ICP``. This will align the digitization points to the
#   head surface. Sometimes the fitting algorithm doesn't find the correct
#   alignment immediately. You can try first fitting using LPA/RPA or fiducials
#   and then align according to the digitization. You can also finetune
#   manually with the controls on the right side of the panel.
# * Click ``Save`` (lower right corner of the panel), set the filename
#   and read it with :func:`mne.read_trans`.
#
# For more information, see this video:
#
# .. youtube:: ALV5qqMHLlQ
#
# .. note::
#     Coregistration can also be automated as shown in :ref:`tut-auto-coreg`.

mne.gui.coregistration(subject="sample", subjects_dir=subjects_dir)


# Coregistration without MRI
# --------------------------
#
# If a template source space, such as 'fsaverage', a sphere, or an infant MRI
# template like 'ANTS4-5Month3T' is loaded as the anatomical model, while
# digitization data from an MEG recording is loaded from the raw file, the
# template source space may need to be warped to match the digitized head
# points.

mne.gui.coregistration(subject="fsaverage", subjects_dir=subjects_dir)

# 'MRI Scaling' menu gives you the options you need to perform warping.
# Select "3-axis" option under the Scaling Mode field in
# the MRI Scaling section. Click ``Fit ICP with scaling`` to warp the template
# to the digitization points.
#
# If you type in the subject name and click ``Save scaled anatomy``, the gui
# will create an anatomical subject file in your subjects directory for the
# subject. Warning: The process of warping all of the MRI surfaces and saving
# can take quite some time, e.g. half an hour or so. The status of the warp
# will be displayed on the bottom bar of the gui window. Don't close the gui
# until the saved files are complete!
