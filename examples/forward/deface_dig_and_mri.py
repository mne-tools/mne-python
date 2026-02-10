"""
.. _deface-dig-and-mri:

================================================
Deface MRI and MEG data for identity protection
================================================

Because facial information can be identifying, subject privacy can be protected
by removing facial information in MEG and MRI data before data sharing.

To learn more about coordinate frames, see :ref:`tut-source-alignment`.
"""

import nibabel as nib
import numpy as np
from scipy import linalg

import mne
from mne.io.constants import FIFF

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


def apply_smoothing(points, tris):
    # TO DO: implement simple smoothing step
    out_points = points
    out_tris = tris
    return out_points, out_tris


def add_head(renderer, points, color, opacity=0.95):
    renderer.mesh(*points.T, triangles=seghead_tri, color=color, opacity=opacity)


# Smooth MRI
# The head surface is stored in "mri" coordinate frame
# (origin at center of volume, units=mm)
seghead_rr, seghead_tri = mne.read_surface(
    subjects_dir / "sample" / "surf" / "lh.seghead"
)

smooth_rr, smooth_tri = apply_smoothing(seghead_rr, seghead_tri)

# The "mri_voxel"→"mri" transform is embedded in the header of the T1 image
# file. We'll invert it and then apply it to the original `seghead_rr` points.
# No unit conversion necessary: this transform expects mm and the scalp surface
# is defined in mm.
vox_to_mri = t1_mgh.header.get_vox2ras_tkr()
mri_to_vox = linalg.inv(vox_to_mri)
scalp_points_in_vox = mne.transforms.apply_trans(mri_to_vox, smooth_rr, move=True)

# Get the nasion:
nasion = [
    p
    for p in raw.info["dig"]
    if p["kind"] == FIFF.FIFFV_POINT_CARDINAL and p["ident"] == FIFF.FIFFV_POINT_NASION
][0]
assert nasion["coord_frame"] == FIFF.FIFFV_COORD_HEAD
nasion = nasion["r"]  # get just the XYZ values
smooth_nasion, _ = apply_smoothing(nasion, None)

# Transform it from head to MRI space (recall that `trans` is head → mri)
nasion_mri = mne.transforms.apply_trans(trans, smooth_nasion, move=True)
# Then transform to voxel space, after converting from meters to millimeters
nasion_vox = mne.transforms.apply_trans(mri_to_vox, nasion_mri * 1e3, move=True)


# Plot smoothed results to make sure the transforms worked
renderer = mne.viz.backends.renderer.create_3d_figure(
    size=(400, 400), bgcolor="w", scene=False
)
add_head(renderer, scalp_points_in_vox, "green", opacity=1)
# plot nasion location
renderer.sphere(center=nasion_vox, color="orange", scale=10)
mne.viz.set_3d_view(
    figure=renderer.figure,
    distance=600.0,
    focalpoint=(0.0, 125.0, 250.0),
    elevation=45,
    azimuth=180,
)
renderer.show()
