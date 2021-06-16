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
Contact locations in MR-space are the goal because this is where
brain structures are represented from the freesurfer reconstruction.
Contact locations in MR-space can lastly be translated to an average space
such as ``fsaverage`` for group comparisons.

.. note::
    The hyperintense skull is actually aligned to the hypointensity between
    the brain and the scalp. The brighter area surrounding the skull in the MR
    is actually subcutaneous fat.

References
----------
.. footbibliography::

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
import mne
from mne.datasets import fetch_fsaverage

print(__doc__)

np.set_printoptions(suppress=True)  # suppress scientific notation

# paths to mne datasets - sample sEEG and FreeSurfer's fsaverage subject
# which is in MNI space
misc_path = mne.datasets.misc.data_path()
sample_path = mne.datasets.sample.data_path()
subject = 'fsaverage'
subjects_dir = sample_path + '/subjects'

# use mne-python's fsaverage data
fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)  # downloads if needed

###############################################################################
# Let's load our T1-weighted image and visualize it. As you can see, the
# image is already aligned to the anterior commissue and posterior commissure
# (ACPC). This is recommended to do before starting. This can be done using
# freesurfer's freeview, e.g.

"""
.. code-block:: bash

    $ freeview $MISC_PATH/seeg/sample_seeg_T1.mgz

"""

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

T1 = nib.freesurfer.load(misc_path + '/seeg/sample_seeg_T1.mgz')
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

###############################################################################
# Let's load our CT image and visualize it with the T1 image.
# As you can see, the CT is already aligned to the T1 image in this example
# dataset.

CT = nib.freesurfer.load(misc_path + '/seeg/sample_seeg_CT.mgz')

# make low intensity parts of the CT transparent for easier visualization
CT_data = CT.get_fdata()
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
for i, ax in enumerate((axes[1], axes[2])):
    ax.annotate('Skull (dark in MR, bright in CT)', (40, 175),
                xytext=(120, 246), horizontalalignment='center',
                color='black' if i == 0 else 'white',
                arrowprops=dict(facecolor='white'))
axes[2].set_title('CT aligned to MR')

###############################################################################
# Let's unalign our CT data so that we can see how to properly align it.

unalign_affine = np.array([
    [0.99833131, -0.02218569, 0.05331175, -1.79438782],
    [0.02443217, 0.99882448, -0.04186315, 6.40490723],
    [-0.05232033, 0.04309582, 0.99769992, 2.46591187],
    [0., 0., 0., 1.]])
CT_unaligned = resample_from_to(CT, (CT.shape, unalign_affine))

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(CT.get_fdata()[128])
axes[1].imshow(CT_unaligned.get_fdata()[128])

reg_img, reg_affine = affine_registration(
    moving=CT_unaligned.get_fdata(),
    static=T1.get_fdata(),
    moving_affine=CT_unaligned.affine,
    static_affine=T1.affine,
    nbins=32,
    metric='MI',
    pipeline=[center_of_mass, translation, rigid, affine],
    level_iters=[100, 100, 10],
    sigmas=[3.0, 1.0, 0.0],
    factors=[4, 2, 1])
CT_aligned = nib.MGHImage(reg_img, reg_affine)
