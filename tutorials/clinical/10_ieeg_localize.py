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
# License: BSD (3-clause)

import os.path as op

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nibabel as nib
from nibabel.processing import resample_from_to
from dipy.align.imaffine import AffineRegistration, MutualInformationMetric
from dipy.align.transforms import RigidTransform3D
from dipy.align import (affine_registration, center_of_mass, translation,
                        rigid, affine)
from dipy.align.reslice import reslice
from dipy.align.metrics import CCMetric
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration

import mne
from mne.datasets import fetch_fsaverage

print(__doc__)

np.set_printoptions(suppress=True)  # suppress scientific notation

# paths to mne datasets - sample sEEG and FreeSurfer's fsaverage subject
# which is in MNI space
misc_path = mne.datasets.misc.data_path()
sample_path = mne.datasets.sample.data_path()
subject = 'fsaverage'
subjects_dir = op.join(sample_path, 'subjects')

# use mne-python's fsaverage data
fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)  # downloads if needed

###############################################################################
# Aligning the T1 to ACPC
# =======================
#
# Let's load our T1-weighted image and visualize it. As you can see, the
# image is already aligned to the anterior commissue and posterior commissure
# (ACPC). This is recommended to do before starting. This can be done using
# freesurfer's freeview:
#
# .. code-block:: bash
#
#     $ freeview $MISC_PATH/seeg/sample_seeg_T1.mgz
#
# And then interact with the graphical user interface:
#
# First, it is recommended to change the cursor style to long, this can be done
# through the menu options like so:
#
#     ``Freeview -> Preferences -> General -> Cursor style -> Long``
#
# Then, the image needs to be aligned to ACPC to look like the image below.
# This can be done by pulling up the transform popup from the menu like so:
#
#     ``Tools -> Transform Volume``
#
# .. note::
#     Be sure to set the text entry box labeled RAS (not TkReg RAS) to
#     ``0 0 0`` before beginning the transform.
#
# Then translate the image until the crosshairs meet on the AC and
# run through the PC as shown in the plot. The eyes should be in
# the ACPC plane and the image should be rotated until they are symmetrical,
# and the crosshairs should transect the midline of the brain.
# Be sure to use both the rotate and the translate menus and save the volume
# after you're finished using ``Save Volume As`` in the transform popup
# :footcite:`HamiltonEtAl2017`.

T1 = nib.freesurfer.load(op.join(misc_path, 'seeg', 'sample_seeg_T1.mgz'))
viewer = T1.orthoview()
viewer.set_position(0, 9.9, 5.8)
viewer.figs[0].axes[0].annotate(
    'PC', (107, 108), xytext=(10, 75), color='white',
    horizontalalignment='center',
    arrowprops=dict(facecolor='white', lw=0.5, width=2, headwidth=5))
viewer.figs[0].axes[0].annotate(
    'AC', (137, 108), xytext=(246, 75), color='white',
    horizontalalignment='center',
    arrowprops=dict(facecolor='white', lw=0.5, width=2, headwidth=5))

###############################################################################
# Freesurfer recon-all
# ====================
#
# Now we're ready for the most time consuming step of the process; the
# freesurfer reconstruction. This process segments out the brain from the
# rest of the MR image and determines which voxels correspond to each brain
# area based on a template deformation. This process takes approximately
#  8 hours so plan accordingly.
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
#     Using the ``--deface`` flag will create a defaced, anonymized T1 image
#     located at ``$MY_DATA_DIRECTORY/$SUBJECT/mri/orig_defaced.mgz``,
#     which is helpful for when you publish your data. You can also use
#     :meth:`mne_bids.write_anat` and pass ``deface=True``.


###############################################################################
# Aligning the CT to the MR
# =========================
#
# Let's load our CT image and visualize it with the T1 image.
# As you can see, the CT is already aligned to the T1 image in this example
# dataset.

CT = nib.freesurfer.load(op.join(misc_path, 'seeg', 'sample_seeg_CT.mgz'))

# make low intensity parts of the CT transparent for easier visualization
CT_data = CT.get_fdata().copy()
CT_data[CT_data < np.quantile(CT_data, 0.95)] = np.nan

fig, axes = plt.subplots(1, 3, figsize=(12, 6))
for ax in axes:
    ax.axis('off')
axes[0].imshow(T1.get_fdata()[T1.shape[0] // 2], cmap='gray')
axes[0].set_title('MR')
axes[1].imshow(CT.get_fdata()[CT.shape[0] // 2], cmap='gray')
axes[1].set_title('CT')
axes[2].imshow(T1.get_fdata()[CT.shape[0] // 2], cmap='gray')
axes[2].imshow(CT_data[CT.shape[0] // 2], cmap='gist_heat', alpha=0.5)
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


###############################################################################
# Let's unalign our CT data so that we can see how to properly align it.
#
# .. note::
#     The hyperintense skull is actually aligned to the hypointensity between
#     the brain and the scalp. The brighter area surrounding the skull in the
#     MR is actually subcutaneous fat.

def plot_overlay(image, compare, title, thresh=None):
    """Define a helper function for comparing plots."""
    image = nib.orientations.apply_orientation(
        image.get_fdata().copy(), nib.orientations.axcodes2ornt(
            nib.orientations.aff2axcodes(image.affine)))
    compare = nib.orientations.apply_orientation(
        compare.get_fdata().copy(), nib.orientations.axcodes2ornt(
            nib.orientations.aff2axcodes(compare.affine)))
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


# Make an affine to transform the image
unalign_affine = np.array([
    [-1.01, 0.02, -0.01, 128],
    [0.01, -0.02, 1.02, -135],
    [0.06, -0.99, -0.02, 140],
    [0, 0, 0, 1]])
CT_unaligned = resample_from_to(CT, (CT.shape, unalign_affine))

plot_overlay(T1, CT_unaligned, 'Unaligned CT Overlaid on T1', thresh=0.95)

###############################################################################
# Now we can align our now unaligned CT image.
#
# .. note::
#     If the alignment fails or takes too long, it is recommended to roughly
#     align the CT to the MR manually. To do this, load both the MR and the CT
#     in freeview and then use the same translation tools as above. Be sure to
#     have the CT image selected in blue in the menu at the top left so that
#     you are adjusting the correct image.
#
#     .. code-block:: bash
#
#         $ freeview $MISC_PATH/seeg/sample_seeg_T1.mgz \
#           $MISC_PATH/seeg/sample_seeg_CT.mgz

affreg = AffineRegistration(
    metric=MutualInformationMetric(nbins=32),
    level_iters=[10], sigmas=[0.0], factors=[1])
rigid_trans = affreg.optimize(
    static=T1.get_fdata(), moving=CT_unaligned.get_fdata(),
    transform=RigidTransform3D(), params0=None,
    static_grid2world=T1.affine, moving_grid2world=CT_unaligned.affine)

trans_affine = np.dot(T1.affine, np.linalg.inv(rigid_trans.affine))
# CT_unaligned = resample(moving=CT_unaligned.get_fdata(), static)
CT_aligned = resample_from_to(CT_unaligned, (CT.shape, trans_affine))

plot_overlay(T1, CT_aligned, 'Aligned CT Overlaid on T1', thresh=0.95)

###############################################################################
# Marking the Location of Each Electrode Contact
# ==============================================
#
# Now, the CT and the MR are in the same space, so when you are looking at a
# point in CT space, it is the same point in MR space. So now everything is
# ready to determine the location of each electrode contact in the
# individual subject's anatomical space (T1-space). To do this, can make
# list of ``RAS`` points from the lower panel in freeview or use the
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
#       $MISC_PATH/seeg/sample_seeg_CT.mgz
#
# Electrode contact locations are plotted below.

# Load electrode positions from file
elec_df = pd.read_csv(op.join(misc_path, 'seeg', 'sample_seeg_electrodes.tsv'),
                      sep='\t', header=0, index_col=None)
ch_names = elec_df['name'].tolist()
ch_coords = elec_df[['R', 'A', 'S']].to_numpy(dtype=float)

# load the subject's brain
subject_brain = nib.freesurfer.load(
    op.join(misc_path, 'seeg', 'sample_seeg_brain.mgz'))

# Make brain surface from T1
verts, triangles = mne.viz.marching_cubes(subject_brain.get_fdata(), level=100)
# transform from voxels to surface RAS
verts = mne.transforms.apply_trans(
    subject_brain.header.get_vox2ras_tkr(), verts)

renderer = mne.viz.backends.renderer.create_3d_figure(
    size=(1200, 900), bgcolor='w', scene=False)
mne.viz.set_3d_view(figure=renderer.figure, distance=700,
                    azimuth=40, elevation=60, focalpoint=(0., 0., -45.))
renderer.mesh(*verts.T, triangles=triangles, color='gray',
              opacity=0.05, representation='surface')
for ch_coord in ch_coords:
    renderer.sphere(center=tuple(ch_coord), color='red', scale=5)

renderer.show()

###############################################################################
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

# load the freesurfer average brain
template_brain = nib.freesurfer.load(
    op.join(subjects_dir, subject, 'mri', 'brain.mgz'))

plot_overlay(template_brain, subject_brain,
             'Alignment with fsaverage before Affine Registration')

###############################################################################
# Now, we'll register the affine of the subject's brain to the template brain.
# This aligns the two brains, preparing the subject's brain to be warped
# to the template.

# normalize intensities
mri_to = template_brain.get_fdata().copy()
mri_to /= mri_to.max()
mri_from = subject_brain.get_fdata().copy()
mri_from /= mri_from.max()

# downsample for speed
zooms = (5, 5, 5)
mri_to, affine_to = reslice(
    mri_to, template_brain.affine,
    template_brain.header.get_zooms()[:3], zooms)
mri_from, affine_from = reslice(
    mri_from, subject_brain.affine,
    subject_brain.header.get_zooms()[:3], zooms)

reg_img, reg_affine = affine_registration(
    moving=mri_from,
    static=mri_to,
    moving_affine=affine_from,
    static_affine=affine_to,
    nbins=32,
    metric='MI',
    pipeline=[center_of_mass, translation, rigid, affine],
    level_iters=[100, 100, 10],
    sigmas=[3.0, 1.0, 0.0],
    factors=[4, 2, 1])

# Apply the transform to the T1 to plot it
aligned_brain = nib.Nifti1Image(reg_img, np.dot(affine_to, reg_affine))
template_brain_zoomed = nib.Nifti1Image(mri_to, affine_to)
plot_overlay(template_brain_zoomed, aligned_brain,
             'Alignment with fsaverage after Affine Registration')

###############################################################################
# Next, we'll compute the symmetric diffeomorphic registration. This accounts
# for differences in the shape and size of the subject's brain areas
# compared to the template brain.

# Compute registration
sdr = SymmetricDiffeomorphicRegistration(
    metric=CCMetric(3), level_iters=[10, 10, 5])
mapping = sdr.optimize(static=template_brain.get_fdata(),
                       moving=subject_brain.get_fdata(),
                       static_grid2world=template_brain.affine,
                       moving_grid2world=subject_brain.affine,
                       prealign=reg_affine)

warped_brain = nib.Nifti1Image(
    mapping.transform(subject_brain.get_fdata()), subject_brain.affine)
plot_overlay(template_brain, warped_brain, 'Warped to fsaverage')

###############################################################################
# Finally, we'll apply the registrations to the electrode contact coordinates.
# The brain image is warped to the template but the goal was to warp the
# positions of the electrode contacts. To do that, we'll make an image that is
# a lookup table of the electrode contacts. In this image, the background will
# be ``0``s all the bright voxels near the location of the first contact will
# be ``1``s, the second ``2``s and so on. This image can then be warped by the
# same SDR transform. We can finally recover a position by averaging the
# positions of all the voxels that had the contact's lookup number in
# the warped image.


def get_neighbors(loc, img, thresh, voxels_in_volume):
    """Find all the neighbors above a threshold near a voxel."""
    neighbors = set()
    for axis in range(len(loc)):
        for i in (-1, 1):
            next_loc = np.array(loc)
            next_loc[axis] += i
            next_loc = tuple(next_loc)
            # must be monotonically decreasing otherwise, bleeds into
            # other contacts
            if img[next_loc] > thresh and img[next_loc] < img[loc] and \
                    next_loc not in voxels_in_volume:
                neighbors.add(next_loc)
    return neighbors


def peak_to_volume(loc, img, thresh, voxels_max=100):
    """Find voxels from peak contact location."""
    loc = tuple(loc)
    voxels_in_volume = neighbors = set([loc])
    while neighbors and len(voxels_in_volume) <= voxels_max:
        next_neighbors = set()
        for next_loc in neighbors:
            voxel_neighbors = get_neighbors(
                next_loc, img, thresh, voxels_in_volume)
            voxels_in_volume = voxels_in_volume.union(voxel_neighbors)
            if len(voxels_in_volume) > voxels_max:
                break
            next_neighbors = next_neighbors.union(voxel_neighbors)
        neighbors = next_neighbors
    return voxels_in_volume


# convert electrode positions from surface RAS to voxels
ch_coords = mne.transforms.apply_trans(
    np.linalg.inv(subject_brain.header.get_vox2ras_tkr()), ch_coords)

# Take channel coordinates and use the CT to transform them
# into a 3D image where all the voxels over a threshold nearby
# are labeled with an index
CT_data = CT.get_fdata()
thresh = np.quantile(CT_data, 0.95)
elec_image = np.zeros(subject_brain.shape, dtype=int)
for i, ch_coord in enumerate(ch_coords):
    x, y, z = ch_coord.round().astype(int)
    # look up to two voxels away, the coord may not have been marked perfectly
    peak = np.array(np.unravel_index(
        np.argmax(CT_data[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]),
        (3, 3, 3))) - 1 + ch_coord.round().astype(int)
    volume = peak_to_volume(peak, CT_data, thresh)
    for voxel in volume:
        if elec_image[voxel] != 0:
            # some voxels ambiguous because the contacts are bridged on the CT
            # so assign the voxel to the nearest contact location
            dist_old = np.sqrt(
                (ch_coords[elec_image[voxel] - 1] - voxel)**2).sum()
            dist_new = np.sqrt((ch_coord - voxel)**2).sum()
            if dist_new < dist_old:
                elec_image[voxel] = i + 1
        else:
            elec_image[voxel] = i + 1

# Apply the mapping
warped_elec_image = mapping.transform(elec_image,
                                      interpolation='nearest')

# Recover the electrode contact positions as the center of mass
for i in range(ch_coords.shape[0]):
    ch_coords[i] = np.array(np.where(warped_elec_image == i + 1)).mean(axis=1)

# Convert back to surface RAS but to the template surface RAS this time
ch_coords = mne.transforms.apply_trans(
    template_brain.header.get_vox2ras_tkr(), ch_coords)

###############################################################################
# We can now plot the result. You can compare this to the plot in
# :ref:`tut-working-with-seeg` to see the difference between this morph which
# is more complex and the less complex linear Talairach transformation.
# By accounting for the shape of this particular subject's brain using the
# SDR to warp the positions of the electrode contacts, the position in the
# template brain is able to be more accurately estimated.

# load electrophysiology data
raw = mne.io.read_raw(op.join(misc_path, 'seeg', 'sample_seeg_ieeg.fif'))

# Create a montage with our new points
ch_pos = dict(zip(ch_names, ch_coords / 1000))  # mm -> m
montage = mne.channels.make_dig_montage(ch_pos, coord_frame='mri_voxel')
raw.set_montage(montage)

# plot the resulting alignment
fig = mne.viz.plot_alignment(raw.info, None, subject,
                             subjects_dir=subjects_dir, show_axes=True,
                             surfaces=['pial', 'head'])

###############################################################################
# References
# ==========
#
# .. footbibliography::
