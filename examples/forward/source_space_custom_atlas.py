"""
.. _ex-source-space-custom-atlas:

=========================================
Source reconstruction with a custom atlas
=========================================

This example shows how to use a custom atlas when performing source reconstruction.
We showcase on the sample dataset how to apply the Yeo atlas during source
 reconstruction.
You should replace the atlas with your own atlas and your own subject.

Any atlas can be used instead of Yeo, provided each region contains a single
 label (ie: no probabilistic atlas).

.. warning:: This tutorial uses FSL and FreeSurfer to perform MRI
 coregistrations. If you use a different software, replace the
 coregistration function appropriately.
"""

# Authors: Fabrice Guibert <fabrice.guibert.96@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%

import subprocess
from pathlib import Path as Path

import nilearn.datasets

import mne
import mne.datasets
from mne._freesurfer import read_freesurfer_lut
from mne.minimum_norm import apply_inverse, make_inverse_operator

# The atlas is in a template space. We download here as an example Yeo
# 2011's atlas, which is in the MNI152 1mm template space.
# Replace this part with your atlas and the template space you used.

nilearn.datasets.fetch_atlas_yeo_2011()  # Download Yeo 2011
yeo_path = Path(
    nilearn.datasets.get_data_dirs()[0], "yeo_2011", "Yeo_JNeurophysiol11_MNI152"
)
atlas_path = Path(yeo_path, "Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz")
atlas_template_T1_path = Path(yeo_path, "FSL_MNI152_FreeSurferConformed_1mm.nii.gz")

# The participant's T1 data. Here, we consider the sample dataset
# The brain should be skull stripped. After freesurfer preprocessing,
# you can either use brain.mgz or antsdn.brain.mgz
data_path = mne.datasets.sample.data_path()
subjects_mri_dir = Path(data_path, "subjects")
subject_mri_path = Path(subjects_mri_dir, "sample")
mri_path = Path(subject_mri_path, "mri")
T1_participant_path = Path(mri_path, "brain.mgz")

assert atlas_path.is_file()
assert atlas_template_T1_path.is_file()
assert T1_participant_path.is_file()

# %%
# The first step is to put the atlas in subject space.
# We show this step with FSL and freesurfer with linear coregistration.
# If your atlas is already in participant space,
# you can skip this step. Coregistration is done in two steps:
# compute the atlas template to subject T1 transform and apply this transform
# to the atlas file with nearest neighbour interpolation.

# FSL does not know how to read .mgz, so we need to convert the T1 to nifti format
# With FreeSurfer:
T1_participant_nifti = Path(str(T1_participant_path).replace("mgz", "nii.gz"))
subprocess.run(["mri_convert", T1_participant_path, T1_participant_nifti])

# Compute template to subject anatomical transform using flirt
template_to_anat_transform_path = Path(mri_path, "template_to_anat.mat")
subprocess.run(
    [
        "flirt",
        "-in",
        atlas_template_T1_path,
        "-ref",
        T1_participant_nifti,
        "-out",
        Path(mri_path, "T1_atlas_coreg"),
        "-omat",
        template_to_anat_transform_path,
    ]
)

# Apply the transform to the atlas
atlas_participant = Path(mri_path, "yeo_atlas.nii.gz")

subprocess.run(
    [
        "flirt",
        "-in",
        atlas_path,
        "-ref",
        T1_participant_nifti,
        "-out",
        atlas_participant,
        "-applyxfm -init",
        template_to_anat_transform_path,
        "-interp nearestneighbour",
    ]
)

# Convert resulting atlas from nifti to mgz
# The filename must finish with aseg, to indicate to MNE that it is
#  a proper atlas segmentation.
atlas_converted = Path(str(atlas_participant).replace(".nii.gz", "aseg.mgz"))
subprocess.run(["mri_convert", atlas_participant, atlas_converted])

assert T1_participant_nifti.is_file()
assert template_to_anat_transform_path.is_file()
assert atlas_participant.is_file()
assert atlas_converted.is_file()

# %%
# With the atlas in participant space, we're still missing one ingredient.
# We need a dictionary mapping label to region ID / value in the fMRI.
# In FreeSurfer and atlases, these typically take the form of lookup tables.
# You can also build the dictionary by hand.

atlas_labels = read_freesurfer_lut(Path(yeo_path, "Yeo2011_7Networks_ColorLUT.txt"))[0]
print(atlas_labels)

# Drop the key corresponding to outer region
del atlas_labels["NONE"]

# %%
# For the purpose of source reconstruction, let's create a volumetric
# source estimate and source reconstruction with e.g eLORETA.
vol_src = mne.setup_volume_source_space(
    "sample",
    subjects_dir=subjects_mri_dir,
    surface=Path(subject_mri_path, "bem", "inner_skull.surf"),
)

fif_path = Path(data_path, "MEG", "sample")
fname_trans = Path(fif_path, "sample_audvis_raw-trans.fif")
raw_fname = Path(fif_path, "sample_audvis_filt-0-40_raw.fif")

model = mne.make_bem_model(
    subject="sample", subjects_dir=subjects_mri_dir, ico=4, conductivity=(0.33,)
)
bem_sol = mne.make_bem_solution(model)

info = mne.io.read_info(raw_fname)
info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False, exclude=[]))

# Build the forward model with our custom source
fwd = mne.make_forward_solution(info, trans=fname_trans, src=vol_src, bem=bem_sol)


# Now perform typical source reconstruction steps
raw = mne.io.read_raw_fif(raw_fname)  # already has an average reference
events = mne.find_events(raw, stim_channel="STI 014")

event_id = dict(aud_l=1)  # event trigger and conditions
tmin = -0.2  # start of each epoch (200ms before the trigger)
tmax = 0.5  # end of each epoch (500ms after the trigger)
raw.info["bads"] = ["MEG 2443", "EEG 053"]
baseline = (None, 0)  # means from the first instant to t = 0
reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)

epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    proj=True,
    picks=("meg", "eog"),
    baseline=baseline,
    reject=reject,
)

# Compute noise covariances
noise_cov = mne.compute_covariance(
    epochs, tmax=0.0, method=["shrunk", "empirical"], rank=None, verbose=True
)

# Compute evoked response
evoked = epochs.average().pick("meg")

# Make inverse operator
inverse_operator = make_inverse_operator(
    evoked.info, fwd, noise_cov, loose=1, depth=0.8
)

# Compute source time courses
method = "eLORETA"
snr = 3.0
lambda2 = 1.0 / snr**2
stc, residual = apply_inverse(
    evoked,
    inverse_operator,
    lambda2,
    method=method,
    pick_ori=None,
    return_residual=True,
    verbose=True,
)

# %%
# Then, we can finally use our atlas!
label_tcs = stc.extract_label_time_course(
    labels=(atlas_converted, atlas_labels), src=vol_src
)
label_tcs.shape
