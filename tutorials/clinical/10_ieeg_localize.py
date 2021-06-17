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
brain structures can be determined using the freesurfer reconstruction
:ref:`tut-freesurfer-reconstruction`. Contact locations in MR-space can also
be translated to an average space such as ``fsaverage`` for group comparisons.
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
from dipy.align import (affine_registration, center_of_mass, translation,
                        rigid, affine)
from dipy.align.metrics import CCMetric
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from skimage import measure
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
#     `0 0 0` before beginning the transform.
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
viewer._axes[0].annotate('PC', (107, 108), xytext=(10, 75),
                         color='white', horizontalalignment='center',
                         arrowprops=dict(facecolor='white', lw=0.5, width=2,
                                         headwidth=5))
viewer._axes[0].annotate('AC', (137, 108), xytext=(246, 75),
                         color='white', horizontalalignment='center',
                         arrowprops=dict(facecolor='white', lw=0.5, width=2,
                                         headwidth=5))

###############################################################################
# Freesurfer recon-all
# ====================
#
# Now we're ready for the most time consuming step of the process; the
# freesurfer reconstruction. This process segments out the brain from the
# rest of the image and determines which voxels correspond to each brain area
# based on a template deformation. This process takes approximately 8 hours
# so it is recommended to plan accordingly.
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
#     to the end of the recon-all command if your scan is not 256 x 256 x 256
#     voxels.
#
# .. note::
#     Using the ``--deface`` flag will create a defaced, anonymized T1 image
#     at the filepath $SUBJECT/mri/orig_defaced.mgz, which is helpful for
#     publishing your data.


###############################################################################
# Aligning the CT to the MR
# =========================
#
# Let's load our CT image and visualize it with the T1 image.
# As you can see, the CT is already aligned to the T1 image in this example
# dataset.
#
# .. note::
#     The hyperintense skull is actually aligned to the hypointensity between
#     the brain and the scalp. The brighter area surrounding the skull in the
#     MR is actually subcutaneous fat.
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

CT = nib.freesurfer.load(op.join(misc_path, 'seeg', 'sample_seeg_CT.mgz'))

# make low intensity parts of the CT transparent for easier visualization
CT_data = CT.get_fdata().copy()
CT_data[CT_data < np.quantile(CT_data, 0.95)] = np.nan

fig, axes = plt.subplots(1, 3, figsize=(12, 6))
for ax in axes:
    ax.axis('off')
axes[0].imshow(T1.get_fdata()[128], cmap='gray')
axes[0].set_title('MR')
axes[1].imshow(CT.get_fdata()[128], cmap='gray')
axes[1].set_title('CT')
axes[2].imshow(T1.get_fdata()[128], cmap='gray')
axes[2].imshow(CT_data[128], cmap='gist_heat', alpha=0.5)
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

# Make an affine slightly different than ``CT.affine`` to transform the image
unalign_affine = np.array([
    [-1.01, 0.02, -0.01, 128],
    [0.01, -0.02, 1.02, -135],
    [0.06, -0.99, -0.02, 140],
    [0, 0, 0, 1]])
CT_unaligned = resample_from_to(CT, (CT.shape, unalign_affine))
CT_data = CT_unaligned.get_fdata().copy()
CT_data[CT_data < np.quantile(CT_data, 0.95)] = np.nan

fig, ax = plt.subplots()
ax.imshow(CT.get_fdata()[128], cmap='gray')
ax.imshow(CT_data[128], cmap='gist_heat', alpha=0.5)
ax.axis('off')
ax.set_title('Unaligned CT Overlaid on Original')

###############################################################################
# Now we can align our now unaligned CT image.

# level iters is set low to increase the speed of execution of the tutorial
# but should be set to [10000, 1000, 100] (default) for actual analyses
reg_img, reg_affine = affine_registration(
    moving=CT_unaligned.get_fdata(),
    static=T1.get_fdata(),
    moving_affine=CT_unaligned.affine,
    static_affine=T1.affine,
    nbins=32,
    metric='MI',
    pipeline=[center_of_mass, translation, rigid, affine],
    level_iters=[100, 10, 5],
    sigmas=[3.0, 1.0, 0.0],
    factors=[4, 2, 1])

CT_aligned = nib.MGHImage(reg_img, reg_affine)
CT_data = CT_aligned.get_fdata().copy()
CT_data[CT_data < np.quantile(CT_data, 0.95)] = np.nan

fig, ax = plt.subplots()
ax.imshow(CT.get_fdata()[128], cmap='gray')
ax.imshow(CT_data[128], cmap='gist_heat', alpha=0.5)
ax.axis('off')
ax.set_title('Aligned CT Overlaid on Original')

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
# the coordinate frame that all the freesurfer outputs are in
# :ref:`tut-tut-freesurfer-mne`.
#
# .. code-block:: bash
#
#     $ freeview $MISC_PATH/seeg/sample_seeg_T1.mgz \
#       $MISC_PATH/seeg/sample_seeg_CT.mgz
#
# Electrode contact locations determined this way are plotted below.

# Load electrode positions from file
elec_df = pd.read_csv(misc_path + '/seeg/sample_seeg_electrodes.tsv',
                      sep='\t', header=0, index_col=None)
ch_names = elec_df['name'].tolist()
ch_coords = elec_df[['R', 'A', 'S']].to_numpy(dtype=float)

# Make brain surface from T1
vert, tri = measure.marching_cubes(T1.get_fdata(), level=100)[:2]
# transform from voxels to surface RAS
vert = mne.transforms.apply_trans(T1.header.get_vox2ras_tkr(), vert)

renderer = mne.viz.backends.renderer.create_3d_figure(
    size=(1200, 900), bgcolor='w', scene=False)
mne.viz.set_3d_view(figure=renderer.figure, distance=700,
                    azimuth=40, elevation=60, focalpoint=(0., 0., -45.))
renderer.mesh(*vert.T, triangles=tri, color='gray',
              opacity=0.05, representation='surface')
for ch_coord in ch_coords:
    renderer.sphere(center=tuple(ch_coord), color='red', scale=5)

renderer.show()

###############################################################################
# Warping to a Common Atlas
# =========================
#
# Electrode contact locations are often compared across subjects in a template
# space such as ``fsaverage`` or ``cvs_avg35_inMNI152``. To transform the
# contact locations to that space, we need to determine a function that maps
# from the T1 to the template space. We will use the symmetric diffeomorphic
# registration (SDR) implemented by ``Dipy`` to do this.
#
# .. note::
#     SDR is more accurate than the linear Talairach transform in
#     :ref:`tut-working-with-seeg` because it allows for non-linear warping.

# load freesurfer average T1 image
fs_T1 = nib.freesurfer.load(
    op.join(subjects_dir, 'fsaverage', 'mri', 'T1.mgz'))

# convert electrode positions from surface RAS to voxels
ch_coords = mne.transforms.apply_trans(
    np.linalg.inv(T1.header.get_vox2ras_tkr()), ch_coords)

# level iters is set low to increase the speed of execution of the tutorial
# but should be set to [10000, 1000, 100] (default) for actual analyses
reg_img, reg_affine = affine_registration(
    moving=T1.get_fdata(),
    static=fs_T1.get_fdata(),
    moving_affine=T1.affine,
    static_affine=fs_T1.affine,
    nbins=32,
    metric='MI',
    pipeline=[center_of_mass, translation, rigid, affine],
    level_iters=[100, 10, 5],
    sigmas=[3.0, 1.0, 0.0],
    factors=[4, 2, 1])

# Compute registration
metric = CCMetric(3)
sdr = SymmetricDiffeomorphicRegistration(metric, level_iters=[100, 10, 5])
mapping = sdr.optimize(static=fs_T1.get_fdata(),
                       moving=T1.get_fdata(),
                       static_grid2world=fs_T1.affine,
                       moving_grid2world=T1.affine,
                       prealign=reg_affine)
# Apply mapping to electrode contact positions
for i, xyz in enumerate(ch_coords):
    x, y, z = np.round(xyz).astype(int)
    ch_coords[i] += mapping.forward[x, y, z]
# convert back to surface RAS but to the template surface RAS this time
ch_coords = mne.transforms.apply_trans(
    fs_T1.header.get_vox2ras_tkr(), ch_coords)

# load electrophysiology data
raw = mne.io.read_raw(misc_path + '/seeg/sample_seeg_ieeg.fif')

# Ideally the nasion/LPA/RPA will also be present from the digitization, here
# we use fiducials estimated from the subject's FreeSurfer MNI transformation:
lpa, nasion, rpa = mne.coreg.get_mni_fiducials(
    subject, subjects_dir=subjects_dir)
lpa, nasion, rpa = lpa['r'], nasion['r'], rpa['r']

# Create a montage with our new points
ch_pos = dict(zip(ch_names, ch_coords))
montage = mne.channels.make_dig_montage(
    ch_pos, coord_frame='mri', nasion=nasion, lpa=lpa, rpa=rpa)
raw.set_montage(montage)

# Get the :term:`trans` that transforms from our MRI coordinate system
# to the head coordinate frame
trans = mne.channels.compute_native_head_t(montage)

# plot the resulting alignment
fig = mne.viz.plot_alignment(raw.info, trans, subject,
                             subjects_dir=subjects_dir, show_axes=True,
                             surfaces=['pial', 'head'])

###############################################################################
# References
# ==========
#
# .. footbibliography::
