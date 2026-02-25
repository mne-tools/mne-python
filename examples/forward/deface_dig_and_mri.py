"""
.. _deface-dig-and-mri:

================================================
Deface MRI and MEG data for identity protection
================================================

Because facial information can be identifying, it is sometimes necessary to
obscure facial detail in MEG and MRI data. This example shows how to do deface
MRI and MEG data without altering head volume or coregistration.

To learn more about coordinate frames, see :ref:`tut-source-alignment`.
"""

import nibabel as nib
import numpy as np
from pyvista import Plotter, PolyData
from scipy import linalg
from scipy.spatial.distance import cdist

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
    surf = PolyData.from_regular_faces(points, tris)
    surf.clean()
    # taubin smoothing conserves volume and triangle face relationships
    smooth_surf = surf.smooth_taubin(n_iter=1000, pass_band=0.0005)
    out_points = smooth_surf.points
    return out_points


def smooth_digitization(dig_p, orig_scalp, smooth_scalp):
    out_dig = dig_p.copy()
    all_dists = cdist(dig_p, orig_scalp)
    closest_point_idxs = all_dists.argmin(axis=1)
    diffs = orig_scalp[closest_point_idxs, :] - smooth_scalp[closest_point_idxs, :]
    for ppi, pp in enumerate(dig_p):
        out_dig[ppi] = pp - diffs[ppi]
    return out_dig


def add_head(renderer, points, tris, color, opacity=0.95):
    renderer.mesh(*points.T, triangles=tris, color=color, opacity=opacity)


# Smooth MRI
# The head surface is stored in "mri" coordinate frame
# (origin at center of volume, units=mm)
seghead_rr, seghead_tri = mne.read_surface(
    subjects_dir / "sample" / "surf" / "lh.seghead"
)

# The "mri_voxel"→"mri" transform is embedded in the header of the T1 image
# file. We'll invert it and then apply it to the original `seghead_rr` points.
# No unit conversion necessary: this transform expects mm and the scalp surface
# is defined in mm.
vox_to_mri = t1_mgh.header.get_vox2ras_tkr()
mri_to_vox = linalg.inv(vox_to_mri)
scalp_points_in_vox = mne.transforms.apply_trans(mri_to_vox, seghead_rr, move=True)


# Get fiducial points and extras:
# "r" selects just the XYZ values
fids = [p["r"] for p in raw.info["dig"] if p["kind"] == FIFF.FIFFV_POINT_CARDINAL]
assert raw.info["dig"][0]["coord_frame"] == FIFF.FIFFV_COORD_HEAD
extra = [e["r"] for e in raw.info["dig"] if e["kind"] == FIFF.FIFFV_POINT_EXTRA]

dig_points = fids + extra
for dpi, dp in enumerate(dig_points):
    # Transform it from head to MRI space (recall that `trans` is head → mri)
    trans_dp = mne.transforms.apply_trans(trans, dp, move=True)
    # Then transform to voxel space, after converting from meters to millimeters
    dig_points[dpi] = mne.transforms.apply_trans(mri_to_vox, trans_dp * 1e3, move=True)

dig_points = np.array(dig_points)

# smooth the whole head

smooth_scalp_points = apply_smoothing(scalp_points_in_vox, seghead_tri)

# The voxel frame origin is located at the top right corner behind the
# subject's head with coordinates in the following order:
# right-to-left axis, superior-to-inferior axis, posterior-to-anterior axis.
# choose facial points from smooth head

fid_y = np.mean([dig_points[0, 2], dig_points[2, 2]])
nasion_z = dig_points[1, 1]

ahead_of_ears = scalp_points_in_vox[:, 2] > fid_y + 10
under_eyebrows = scalp_points_in_vox[:, 1] > nasion_z - 15
idxs_to_smooth = np.where(ahead_of_ears & under_eyebrows)[0]

tris_to_smooth = np.isin(seghead_tri, idxs_to_smooth).all(axis=1)

# choose dig to smooth
dig_ahead_of_ears = dig_points[:, 2] > fid_y + 10
dig_under_brows = dig_points[:, 1] > nasion_z - 15
dig_to_smooth = np.where(dig_ahead_of_ears & dig_under_brows)[0]

smooth_dig = smooth_digitization(
    dig_points[dig_to_smooth], scalp_points_in_vox, smooth_scalp_points
)

# preview point selection from face and facial dig
preview = Plotter()

preview.add_mesh(
    smooth_dig,
    color="red",
    render_points_as_spheres=True,
    point_size=8,
)

preview.add_mesh(
    smooth_scalp_points[idxs_to_smooth, :],
    color="blue",
    render_points_as_spheres=True,
    point_size=2,
)
preview.show()
preview.close()

# Plot smoothed results to make sure the transforms worked
renderer = mne.viz.backends.renderer.create_3d_figure(
    size=(400, 400), bgcolor="w", scene=False
)

original_head_surf = scalp_points_in_vox.copy()

defaced_surf = scalp_points_in_vox.copy()
defaced_surf[idxs_to_smooth, :] = smooth_scalp_points[idxs_to_smooth, :]

defaced_dig = dig_points.copy()
defaced_dig[dig_to_smooth] = smooth_dig

# plot original head surface in grey
add_head(renderer, original_head_surf, seghead_tri, "grey", opacity=0.3)

add_head(renderer, defaced_surf, seghead_tri, "green", opacity=0.5)

# plot fiducials
renderer.sphere(center=defaced_dig[0], color="orange", scale=10)
renderer.sphere(center=defaced_dig[1], color="orange", scale=10)
renderer.sphere(center=defaced_dig[2], color="orange", scale=10)
# plot dig points
for sd in defaced_dig[3:]:
    renderer.sphere(center=sd, color="red", scale=3)
# show the plot
mne.viz.set_3d_view(
    figure=renderer.figure,
    distance=600.0,
    focalpoint=(0.0, 125.0, 250.0),
    elevation=45,
    azimuth=180,
)
renderer.show()
