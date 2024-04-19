"""
.. _tut-working-with-seeg:

======================
Working with sEEG data
======================

MNE-Python supports working with more than just MEG and EEG data. Here we show
some of the functions that can be used to facilitate working with
stereoelectroencephalography (sEEG) data.

This example shows how to use:

- sEEG data
- channel locations in MNI space
- projection into a volume

Note that our sample sEEG electrodes are already assumed to be in MNI
space. If you want to map positions from your subject MRI space to MNI
fsaverage space, you must apply the FreeSurfer's talairach.xfm transform
for your dataset. You can take a look at :ref:`tut-freesurfer-mne` for more
information.

For an example that involves ECoG data, channel locations in a
subject-specific MRI, or projection into a surface, see
:ref:`tut-working-with-ecog`. In the ECoG example, we show
how to visualize surface grid channels on the brain.

Please note that this tutorial requires 3D plotting dependencies,
see :ref:`manual-install`.
"""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Adam Li <adam2392@gmail.com>
#          Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

import dipy.reconst.dti as dti
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.denoise.gibbs import gibbs_removal
from dipy.denoise.patch2self import patch2self
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.direction.peaks import peaks_from_model
from dipy.segment.mask import median_otsu
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.tracking.utils import seeds_from_mask

import mne
from mne.datasets import fetch_fsaverage

# paths to mne datasets - sample sEEG and FreeSurfer's fsaverage subject
# which is in MNI space
misc_path = mne.datasets.misc.data_path()
sample_path = mne.datasets.sample.data_path()
subjects_dir = sample_path / "subjects"

# use mne-python's fsaverage data
fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)  # downloads if needed

# %%
# Let's load some sEEG data with channel locations and make epochs.

raw = mne.io.read_raw(misc_path / "seeg" / "sample_seeg_ieeg.fif")

epochs = mne.Epochs(raw, detrend=1, baseline=None)
epochs = epochs["Response"][0]  # just process one epoch of data for speed

# %%
# Let use the Talairach transform computed in the Freesurfer recon-all
# to apply the Freesurfer surface RAS ('mri') to MNI ('mni_tal') transform.

montage = epochs.get_montage()

# first we need a head to mri transform since the data is stored in "head"
# coordinates, let's load the mri to head transform and invert it
this_subject_dir = misc_path / "seeg"
head_mri_t = mne.coreg.estimate_head_mri_t("sample_seeg", this_subject_dir)
# apply the transform to our montage
montage.apply_trans(head_mri_t)

# now let's load our Talairach transform and apply it
mri_mni_t = mne.read_talxfm("sample_seeg", misc_path / "seeg")
montage.apply_trans(mri_mni_t)  # mri to mni_tal (MNI Taliarach)

# for fsaverage, "mri" and "mni_tal" are equivalent and, since
# we want to plot in fsaverage "mri" space, we need use an identity
# transform to equate these coordinate frames
montage.apply_trans(mne.transforms.Transform(fro="mni_tal", to="mri", trans=np.eye(4)))

epochs.set_montage(montage)

# %%
# Let's check to make sure everything is aligned.
#
# .. note::
#    The most rostral electrode in the temporal lobe is outside the
#    fsaverage template brain. This is not ideal but it is the best that
#    the linear Talairach transform can accomplish. A more complex
#    transform is necessary for more accurate warping, see
#    :ref:`tut-ieeg-localize`.

# compute the transform to head for plotting
trans = mne.channels.compute_native_head_t(montage)
# note that this is the same as:
# ``mne.transforms.invert_transform(
#      mne.transforms.combine_transforms(head_mri_t, mri_mni_t))``

view_kwargs = dict(azimuth=105, elevation=100, focalpoint=(0, 0, -15))
brain = mne.viz.Brain(
    "fsaverage",
    subjects_dir=subjects_dir,
    cortex="low_contrast",
    alpha=0.25,
    background="white",
)
brain.add_sensors(epochs.info, trans=trans)
brain.add_head(alpha=0.25, color="tan")
brain.show_view(distance=400, **view_kwargs)

# %%
# Now, let's project onto the inflated brain surface for visualization.
# This video may be helpful for understanding the how the annotations on
# the pial surface translate to the inflated brain and flat map:
#
# .. youtube:: OOy7t1yq8IM
brain = mne.viz.Brain(
    "fsaverage", subjects_dir=subjects_dir, surf="inflated", background="black"
)
brain.add_annotation("aparc.a2009s")
brain.add_sensors(epochs.info, trans=trans)
brain.show_view(distance=500, **view_kwargs)

# %%
# Let's also show the sensors on a flat brain.
brain = mne.viz.Brain(
    "fsaverage", subjects_dir=subjects_dir, surf="flat", background="black"
)
brain.add_annotation("aparc.a2009s")
brain.add_sensors(epochs.info, trans=trans)

# %%
# Let's also look at which regions of interest are nearby our electrode
# contacts.

aseg = "aparc+aseg"  # parcellation/anatomical segmentation atlas
labels, colors = mne.get_montage_volume_labels(
    montage, "fsaverage", subjects_dir=subjects_dir, aseg=aseg
)

# separate by electrodes which have names like LAMY 1
electrodes = set(
    [
        "".join([lttr for lttr in ch_name if not lttr.isdigit() and lttr != " "])
        for ch_name in montage.ch_names
    ]
)
print(f"Electrodes in the dataset: {electrodes}")

electrodes = ("LPM", "LSMA")  # choose two for this example
for elec in electrodes:
    picks = [ch_name for ch_name in epochs.ch_names if elec in ch_name]
    fig, ax = mne.viz.plot_channel_labels_circle(labels, colors, picks=picks)
    fig.text(0.3, 0.9, "Anatomical Labels", color="white")

# %%
# For electrode contacts in white matter, it can be helpful to visualize
# fiber tracts that pass nearby as well. For that we need to do fiber
# tracking on diffusion MR data.

# load the diffusion MR data
dwi = nib.load(misc_path / "seeg" / "sample_seeg_dwi.nii.gz")
bvals = np.loadtxt(misc_path / "seeg" / "sample_seeg_dwi.bval")
bvecs = np.loadtxt(misc_path / "seeg" / "sample_seeg_dwi.bvec")
gtab = gradient_table(bvals, bvecs)

# use B0 diffusion data to align with the T1
b0_idx = tuple(np.where(bvals < 50)[0])
dwi_masked, mask = median_otsu(np.array(dwi.dataobj), vol_idx=b0_idx)

fig, ax = plt.subplots()
ax.imshow(np.rot90(dwi_masked[65, ..., 0]), aspect="auto")

t1 = nib.load(misc_path / "seeg" / "sample_seeg" / "mri", "T1.mgz")
dwi_b0_register = nib.Nifti1Image(dwi_masked[..., b0_idx].mean(axis=-1), dwi.affine)

# %%
# The code below was run once to find the registration matrix, but to
# save computer resources when building the documentation, we won't
# run it every time::
#
# reg_affine = mne.transforms.compute_volume_registration(
#     moving=dwi_b0_register, static=t1, pipeline='rigids')

reg_affine = np.array(
    [
        [0.99804908, -0.05071631, 0.03641263, 1.36631239],
        [0.049687, 0.99835418, 0.0286378, 36.79845134],
        [-0.03780511, -0.02677269, 0.99892642, 8.30634414],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
reg_affine_inv = np.linalg.inv(reg_affine)

# use registration to move the white matter mask computed
# by freesurfer to the diffusion space
wm = nib.load(misc_path / "seeg" / "sample_seeg" / "mri" / "wm.mgz")
wm_data = np.array(wm.dataobj)
wm_mask = (wm_data == 109) | (wm_data == 110)  # white matter values
wm = nib.MGHImage(wm_mask.astype(np.float32), wm.affine)
del wm_data, wm_mask

# apply the backward registration by using the inverse
wm_dwi = mne.transforms.apply_volume_registration(
    moving=wm, static=dwi_b0_register, reg_affine=reg_affine_inv
)

# check that white matter is aligned properly
fig, ax = plt.subplots()
ax.imshow(np.rot90(dwi_b0_register.dataobj[56]), aspect="auto")
ax.imshow(np.rot90(wm_dwi.dataobj[56]), aspect="auto", cmap="hot", alpha=0.5)

# now, preprocess the diffusion data to remove noise and do
# fiber tracking
denoised = patch2self(dwi_masked, bvals)
denoised = gibbs_removal(denoised)

# %%
# You may also want to do the following, but it registers each direction
# of the diffusion image to the T1, so it takes a lot of computational
# resources so we'll skip it for now::
#
# from dipy.align import motion_correction
# denoised = motion_correction(denoised, dwi.affine, b0_ref=0)

# compute diffusion tensor imaging to find the peak direction
# for each voxel
tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(denoised)
pam = peaks_from_model(
    tenmodel,
    denoised,
    default_sphere,
    relative_peak_threshold=0.5,
    min_separation_angle=25,
    mask=wm_dwi.dataobj,
)

# do fiber tracking
stopping_criterion = ThresholdStoppingCriterion(
    pam.gfa,  # use generalized fractional anisotropy from the DTI model
    0.25,  # threshold for stopping is when FA goes below 0.25 (default)
)
dg = DeterministicMaximumDirectionGetter.from_shcoeff(
    pam.shm_coeff,  # use spherical harmonic coefficients from the DTI model
    max_angle=30.0,  # max angle fiber can change at each voxel
    sphere=default_sphere,  # use default sphere
    sh_to_pmf=True,  # speeds up computations, takes more memory
)
# use the white matter mask to seed where the fibers start,
# with 1 mm density in all three dimensions
seeds = seeds_from_mask(wm_dwi.dataobj, dwi.affine, density=(1, 1, 1))
# generate streamlines to represent tracts using the stopping
# criteria, direction getter and seeds
streamline_generator = LocalTracking(
    dg, stopping_criterion, seeds, dwi.affine, step_size=0.5
)
streamlines = Streamlines(streamline_generator)

# move streamlines from diffusion space to T1 anatomical space,
# only keep non-singleton streamlines
streamlines = [
    mne.transforms.apply_trans(reg_affine_inv, streamline)
    for streamline in streamlines
    if len(streamline) > 1
]

# now convert from scanner RAS to surface RAS
ras2mri = mne.transforms.combine_transforms(
    mne.transforms.Transform("ras", "mri_voxel", t1.header.get_ras2vox()),
    mne.transforms.Transform("mri_voxel", "mri", t1.header.get_vox2ras_tkr()),
    fro="ras",
    to="mri",
)
streamlines = [
    mne.transforms.apply_trans(ras2mri, streamline) / 1000  # mm -> m
    for streamline in streamlines
]

# %%
# Now, let's the electrodes and a few regions of interest that the contacts
# of the electrode are proximal to.

picks = [
    ii
    for ii, ch_name in enumerate(epochs.ch_names)
    if any([elec in ch_name for elec in electrodes])
]
labels = (
    "ctx-lh-caudalmiddlefrontal",
    "ctx-lh-precentral",
    "ctx-lh-superiorfrontal",
    "Left-Putamen",
)

fig = mne.viz.plot_alignment(
    mne.pick_info(epochs.info, picks),
    trans,
    "fsaverage",
    subjects_dir=subjects_dir,
    surfaces=[],
    coord_frame="mri",
)

brain = mne.viz.Brain(
    "fsaverage",
    alpha=0.1,
    cortex="low_contrast",
    subjects_dir=subjects_dir,
    units="m",
    figure=fig,
)
brain.add_volume_labels(aseg="aparc+aseg", labels=labels)

# find streamlines near LSMA1
montage = epochs.get_montage()
montage.apply_trans(mne.transforms.invert_transform(trans))  # head -> mri
ch_pos = montage.get_positions()["ch_pos"]

thresh = 0.05  # pick streamlines within 3 mm
streamlines_pick = [
    streamline
    for streamline in streamlines
    if np.linalg.norm(streamline - ch_pos["LPM 1"]).min() < thresh
]

brain.add_streamlines(streamlines_pick, color="white")

brain.show_view(azimuth=120, elevation=90, distance=0.25)

# %%
# Next, we'll get the epoch data and plot its amplitude over time.

epochs.plot(events=True)

# %%
# We can visualize this raw data on the ``fsaverage`` brain (in MNI space) as
# a heatmap. This works by first creating an ``Evoked`` data structure
# from the data of interest (in this example, it is just the raw LFP).
# Then one should generate a ``stc`` data structure, which will be able
# to visualize source activity on the brain in various different formats.

# get standard fsaverage volume (5mm grid) source space
fname_src = subjects_dir / "fsaverage" / "bem" / "fsaverage-vol-5-src.fif"
vol_src = mne.read_source_spaces(fname_src)

evoked = epochs.average()
stc = mne.stc_near_sensors(
    evoked,
    trans,
    "fsaverage",
    subjects_dir=subjects_dir,
    src=vol_src,
    surface=None,
    verbose="error",
)
stc = abs(stc)  # just look at magnitude
clim = dict(kind="value", lims=np.percentile(abs(evoked.data), [10, 50, 75]))

# %%
# Plot 3D source (brain region) visualization:
#
# By default, `stc.plot_3d() <mne.VolSourceEstimate.plot_3d>` will show a time
# course of the source with the largest absolute value across any time point.
# In this example, it is simply the source with the largest raw signal value.
# Its location is marked on the brain by a small blue sphere.

# sphinx_gallery_thumbnail_number = 6

brain = stc.plot_3d(
    src=vol_src,
    subjects_dir=subjects_dir,
    view_layout="horizontal",
    views=["axial", "coronal", "sagittal"],
    size=(800, 300),
    show_traces=0.4,
    clim=clim,
    add_data_kwargs=dict(colorbar_kwargs=dict(label_font_size=8)),
)

# You can save a movie like the one on our documentation website with:
# brain.save_movie(time_dilation=3, interpolation='linear', framerate=5,
#                  time_viewer=True, filename='./mne-test-seeg.m4')

# %%
# In this tutorial, we used a BEM surface for the ``fsaverage`` subject from
# FreeSurfer.
#
# For additional common analyses of interest, see the following:
#
# - For volumetric plotting options, including limiting to a specific area of
#   the volume specified by say an atlas, or plotting different types of
#   source visualizations see:
#   :ref:`tut-viz-stcs`.
# - For extracting activation within a specific FreeSurfer volume and using
#   different FreeSurfer volumes, see: :ref:`tut-freesurfer-mne`.
# - For working with BEM surfaces and using FreeSurfer, or MNE to generate
#   them, see: :ref:`tut-forward`.
