# -*- coding: utf-8 -*-
"""
.. _tut-ieeg-localize:

========================================
Locating Intracranial Electrode Contacts
========================================

Intracranial electrophysiology recording contacts are generally localized
based on a post-implantation computed tomography (CT) image and a
pre-implantation magnetic resonance (MR) image. The CT image has greater
intensity than the background at each of the electrode contacts and
for the skull. Using the skull, the CT can be aligned to MR-space.
Contact locations in MR-space are the goal because this is the image from which
brain structures can be determined using the
:ref:`tut-freesurfer-reconstruction`. Contact locations in MR-space can also
be translated to a template space such as ``fsaverage`` for group comparisons.
"""

# Authors: Alex Rockhill <aprockhill@mailbox.org>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

# %%

import os
import os.path as op

import numpy as np
import matplotlib.pyplot as plt

import nibabel as nib
import nilearn.plotting
from dipy.align import resample

import mne
from mne.datasets import fetch_fsaverage

# paths to mne datasets - sample sEEG and FreeSurfer's fsaverage subject
# which is in MNI space
misc_path = mne.datasets.misc.data_path()
sample_path = mne.datasets.sample.data_path()
subjects_dir = op.join(sample_path, 'subjects')

# use mne-python's fsaverage data
fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)  # downloads if needed

# %%
# Freesurfer recon-all
# ====================
#
# The first step is the most time consuming; the freesurfer reconstruction.
# This process segments out the brain from the rest of the MR image and
# determines which voxels correspond to each brain area based on a template
# deformation. This process takes approximately 8 hours so plan accordingly.
#
# .. code-block:: bash
#
#     $ export SUBJECT=sample_seeg
#     $ export SUBJECTS_DIR=$MY_DATA_DIRECTORY
#     $ recon-all -subjid $SUBJECT -sd $SUBJECTS_DIR \
#       -i $MISC_PATH/seeg/sample_seeg_T1.mgz -all -deface
#
# .. note::
#     You may need to include an additional ``-cw256`` flag which can be added
#     to the end of the recon-all command if your MR scan is not
#     ``256 x 256 x 256`` voxels.
#
# .. note::
#     Using the ``-deface`` flag will create a defaced, anonymized T1 image
#     located at ``$MY_DATA_DIRECTORY/$SUBJECT/mri/orig_defaced.mgz``,
#     which is helpful for when you publish your data. You can also use
#     :func:`mne_bids.write_anat` and pass ``deface=True``.


# %%
# Aligning the CT to the MR
# =========================
#
# Let's load our T1 and CT images and visualize them. You can hardly
# see the CT, it's so misaligned that it is mostly out of view but there is a
# part of the skull upsidedown and way off center in the middle plot.
# Clearly, we need to align the CT to the T1 image.

def plot_overlay(image, compare, title, thresh=None):
    """Define a helper function for comparing plots."""
    image = nib.orientations.apply_orientation(
        np.asarray(image.dataobj), nib.orientations.axcodes2ornt(
            nib.orientations.aff2axcodes(image.affine))).astype(np.float32)
    compare = nib.orientations.apply_orientation(
        np.asarray(compare.dataobj), nib.orientations.axcodes2ornt(
            nib.orientations.aff2axcodes(compare.affine))).astype(np.float32)
    if thresh is not None:
        compare[compare < np.quantile(compare, thresh)] = np.nan
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title)
    for i, ax in enumerate(axes):
        ax.imshow(np.take(image, [image.shape[i] // 2], axis=i).squeeze().T,
                  cmap='gray')
        ax.imshow(np.take(compare, [compare.shape[i] // 2],
                          axis=i).squeeze().T, cmap='gist_heat', alpha=0.5)
        ax.invert_yaxis()
        ax.axis('off')
    fig.tight_layout()


T1 = nib.load(op.join(misc_path, 'seeg', 'sample_seeg', 'mri', 'T1.mgz'))
CT_orig = nib.load(op.join(misc_path, 'seeg', 'sample_seeg_CT.mgz'))

# resample to T1's definition of world coordinates
CT_resampled = resample(moving=np.asarray(CT_orig.dataobj),
                        static=np.asarray(T1.dataobj),
                        moving_affine=CT_orig.affine,
                        static_affine=T1.affine)
plot_overlay(T1, CT_resampled, 'Unaligned CT Overlaid on T1', thresh=0.95)
del CT_resampled

# %%
# Now we need to align our CT image to the T1 image.
#
# We want this to be a rigid transformation (just rotation + translation),
# so we don't do a full affine registration (that includes shear) here.
# This takes a while (~10 minutes) to execute so we skip actually running it
# here::
#
#    reg_affine, _ = mne.transforms.compute_volume_registration(
#        CT_orig, T1, pipeline='rigids', verbose=True)
#
# And instead we just hard-code the resulting 4x4 matrix:

reg_affine = np.array([
    [0.99270756, -0.03243313, 0.11610254, -133.094156],
    [0.04374389, 0.99439665, -0.09623816, -97.58320673],
    [-0.11233068, 0.10061512, 0.98856381, -84.45551601],
    [0., 0., 0., 1.]])
CT_aligned = mne.transforms.apply_volume_registration(CT_orig, T1, reg_affine)
plot_overlay(T1, CT_aligned, 'Aligned CT Overlaid on T1', thresh=0.95)
del CT_orig

# %%
# We can now see how the CT image looks properly aligned to the T1 image.
#
# .. note::
#     The hyperintense skull is actually aligned to the hypointensity between
#     the brain and the scalp. The brighter area surrounding the skull in the
#     MR is actually subcutaneous fat.

# make low intensity parts of the CT transparent for easier visualization
CT_data = CT_aligned.get_fdata().copy()
CT_data[CT_data < np.quantile(CT_data, 0.95)] = np.nan
T1_data = np.asarray(T1.dataobj)

fig, axes = plt.subplots(1, 3, figsize=(12, 6))
for ax in axes:
    ax.axis('off')
axes[0].imshow(T1_data[T1.shape[0] // 2], cmap='gray')
axes[0].set_title('MR')
axes[1].imshow(np.asarray(CT_aligned.dataobj)[CT_aligned.shape[0] // 2],
               cmap='gray')
axes[1].set_title('CT')
axes[2].imshow(T1_data[T1.shape[0] // 2], cmap='gray')
axes[2].imshow(CT_data[CT_aligned.shape[0] // 2], cmap='gist_heat', alpha=0.5)
for ax in (axes[0], axes[2]):
    ax.annotate('Subcutaneous fat', (110, 52), xytext=(100, 30),
                color='white', horizontalalignment='center',
                arrowprops=dict(facecolor='white'))
for ax in axes:
    ax.annotate('Skull (dark in MR, bright in CT)', (40, 175),
                xytext=(120, 246), horizontalalignment='center',
                color='white', arrowprops=dict(facecolor='white'))
axes[2].set_title('CT aligned to MR')
fig.tight_layout()
del CT_data, T1

# %%
# Marking the Location of Each Electrode Contact
# ==============================================
#
# Now, the CT and the MR are in the same space, so when you are looking at a
# point in CT space, it is the same point in MR space. So now everything is
# ready to determine the location of each electrode contact in the
# individual subject's anatomical space (T1-space). To do this, can make
# list of ``TkReg RAS`` points from the lower panel in freeview or use the
# mne graphical user interface (coming soon). The electrode locations will then
# be in the ``surface RAS`` coordinate frame, which is helpful because that is
# the coordinate frame that all the surface and image files that freesurfer
# outputs are in, see :ref:`tut-freesurfer-mne`.
#
# The electrode contact locations could be determined using freeview by
# clicking through and noting each contact position in the interface launched
# by the following command:
#
# .. code-block:: bash
#
#     $ freeview $MISC_PATH/seeg/sample_seeg_T1.mgz \
#           $MISC_PATH/seeg/sample_seeg_CT.mgz

# %%
# Now, we'll make a montage with the channels that we've found in the
# previous step.
#
# .. note:: MNE represents data in the "head" space internally

# load electrophysiology data with channel locations
raw = mne.io.read_raw(op.join(misc_path, 'seeg', 'sample_seeg_ieeg.fif'))

# create symbolic link to share ``subjects_dir``
if not op.exists(op.join(subjects_dir, 'sample_seeg')):
    os.symlink(op.join(misc_path, 'seeg', 'sample_seeg'),
               op.join(subjects_dir, 'sample_seeg'))

# %%
# Let's plot the electrode contact locations on the subject's brain.
#
# MNE stores digitization montages in a coordinate frame called "head"
# defined by fiducial points (origin is halfway between the LPA and RPA
# see :ref:`tut-source-alignment`). For sEEG, it is convenient to get an
# estimate of the location of the fiducial points for the subject
# using the Talairach transform (see :func:`mne.coreg.get_mni_fiducials`)
# to use to define the coordinate frame so that we don't have to manually
# identify their location. The estimated head->mri ``trans`` was used
# when the electrode contacts were localized so we need to use it again here.

# estimate head->mri transform
subj_trans = mne.coreg.estimate_head_mri_t('sample_seeg', subjects_dir)

# plot the alignment
brain_kwargs = dict(cortex='low_contrast', alpha=0.2, background='white',
                    subjects_dir=subjects_dir)
brain = mne.viz.Brain('sample_seeg', **brain_kwargs)
brain.add_sensors(raw.info, trans=subj_trans)
view_kwargs = dict(azimuth=60, elevation=100, distance=350,
                   focalpoint=(0, 0, -15))
brain.show_view(**view_kwargs)

# %%
# Warping to a Common Atlas
# =========================
#
# Electrode contact locations are often compared across subjects in a template
# space such as ``fsaverage`` or ``cvs_avg35_inMNI152``. To transform electrode
# contact locations to that space, we need to determine a function that maps
# from the subject's brain to the template brain. We will use the symmetric
# diffeomorphic registration (SDR) implemented by ``Dipy`` to do this.
#
# Before we can make a function to account for individual differences in the
# shape and size of brain areas, we need to fix the alignment of the brains.
# The plot below shows that they are not yet aligned.

# load the subject's brain and the Freesurfer "fsaverage" template brain
subject_brain = nib.load(
    op.join(misc_path, 'seeg', 'sample_seeg', 'mri', 'brain.mgz'))
template_brain = nib.load(
    op.join(subjects_dir, 'fsaverage', 'mri', 'brain.mgz'))

plot_overlay(template_brain, subject_brain,
             'Alignment with fsaverage before Affine Registration')

# %%
# Now, we'll register the affine of the subject's brain to the template brain.
# This aligns the two brains, preparing the subject's brain to be warped
# to the template.
#
# .. warning:: Here we use ``zooms=5`` just for speed, in general we recommend
#              using ``zooms=None``` (default) for highest accuracy. To deal
#              with this coarseness, we also use a threshold of 0.8 for the CT
#              electrodes rather than 0.95. This coarse zoom and low threshold
#              is useful for getting a quick view of the data, but finalized
#              pipelines should use ``zooms=None`` instead!

CT_thresh = 0.8  # 0.95 is better for zooms=None!
reg_affine, sdr_morph = mne.transforms.compute_volume_registration(
    subject_brain, template_brain, zooms=5, verbose=True)
subject_brain_sdr = mne.transforms.apply_volume_registration(
    subject_brain, template_brain, reg_affine, sdr_morph)

# apply the transform to the subject brain to plot it
plot_overlay(template_brain, subject_brain_sdr,
             'Alignment with fsaverage after SDR Registration')

del subject_brain, template_brain

# %%
# Finally, we'll apply the registrations to the electrode contact coordinates.
# The brain image is warped to the template but the goal was to warp the
# positions of the electrode contacts. To do that, we'll make an image that is
# a lookup table of the electrode contacts. In this image, the background will
# be ``0`` s all the bright voxels near the location of the first contact will
# be ``1`` s, the second ``2`` s and so on. This image can then be warped by
# the SDR transform. We can finally recover a position by averaging the
# positions of all the voxels that had the contact's lookup number in
# the warped image.

# first we need our montage but it needs to be converted to "mri" coordinates
# using our ``subj_trans``
montage = raw.get_montage()
montage.apply_trans(subj_trans)

montage_warped, elec_image, warped_elec_image = mne.warp_montage_volume(
    montage, CT_aligned, reg_affine, sdr_morph,
    subject_from='sample_seeg', subjects_dir=subjects_dir, thresh=CT_thresh)

fig, axes = plt.subplots(2, 1, figsize=(8, 8))
nilearn.plotting.plot_glass_brain(elec_image, axes=axes[0], cmap='Dark2')
fig.text(0.1, 0.65, 'Subject T1', rotation='vertical')
nilearn.plotting.plot_glass_brain(warped_elec_image, axes=axes[1],
                                  cmap='Dark2')
fig.text(0.1, 0.25, 'fsaverage', rotation='vertical')
fig.suptitle('Electrodes warped to fsaverage')

del CT_aligned

# %%
# We can now plot the result. You can compare this to the plot in
# :ref:`tut-working-with-seeg` to see the difference between this morph, which
# is more complex, and the less-complex, linear Talairach transformation.
# By accounting for the shape of this particular subject's brain using the
# SDR to warp the positions of the electrode contacts, the position in the
# template brain is able to be more accurately estimated.

# sphinx_gallery_thumbnail_number = 8

# first we need to add fiducials so that we can define the "head" coordinate
# frame in terms of them (with the origin at the center between LPA and RPA)
montage_warped.add_estimated_fiducials('fsaverage', subjects_dir)

# compute the head<->mri ``trans`` now using the fiducials
template_trans = mne.channels.compute_native_head_t(montage_warped)

# now we can set the montage and, because there are fiducials in the montage,
# the montage will be properly transformed to "head" coordinates when we do
# (this step uses ``template_trans`` but it is recomputed behind the scenes)
raw.set_montage(montage_warped)

# plot the resulting alignment
brain = mne.viz.Brain('fsaverage', **brain_kwargs)
brain.add_sensors(raw.info, trans=template_trans)
brain.show_view(**view_kwargs)

# %%
# This pipeline was developed based on previous work
# :footcite:`HamiltonEtAl2017`.

# %%
# References
# ==========
#
# .. footbibliography::
