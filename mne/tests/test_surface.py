# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from mne import (
    decimate_surface,
    dig_mri_distances,
    get_montage_volume_labels,
    pick_types,
    read_surface,
    write_surface,
)
from mne._fiff.constants import FIFF
from mne.channels import make_dig_montage
from mne.datasets import testing
from mne.io import read_info
from mne.surface import (
    _compute_nearest,
    _get_ico_surface,
    _marching_cubes,
    _normal_orth,
    _project_onto_surface,
    _read_patch,
    _tessellate_sphere,
    _voxel_neighbors,
    fast_cross_3d,
    get_head_surf,
    get_meg_helmet_surf,
    read_curvature,
)
from mne.transforms import _get_trans
from mne.utils import _record_warnings, catch_logging, object_diff, requires_freesurfer

data_path = testing.data_path(download=False)
subjects_dir = data_path / "subjects"
fname = subjects_dir / "sample" / "bem" / "sample-1280-1280-1280-bem-sol.fif"
fname_trans = data_path / "MEG" / "sample" / "sample_audvis_trunc-trans.fif"
fname_raw = data_path / "MEG" / "sample" / "sample_audvis_trunc_raw.fif"
fname_t1 = subjects_dir / "fsaverage" / "mri" / "T1.mgz"
rng = np.random.RandomState(0)


def test_helmet():
    """Test loading helmet surfaces."""
    base_dir = Path(__file__).parents[1] / "io"
    fname_raw = base_dir / "tests" / "data" / "test_raw.fif"
    fname_kit_raw = base_dir / "kit" / "tests" / "data" / "test_bin_raw.fif"
    fname_bti_raw = base_dir / "bti" / "tests" / "data" / "exported4D_linux_raw.fif"
    fname_ctf_raw = base_dir / "tests" / "data" / "test_ctf_raw.fif"
    fname_trans = base_dir / "tests" / "data" / "sample-audvis-raw-trans.txt"
    trans = _get_trans(fname_trans)[0]
    new_info = read_info(fname_raw)
    artemis_info = new_info.copy()
    for pick in pick_types(new_info, meg=True):
        new_info["chs"][pick]["coil_type"] = 9999
        artemis_info["chs"][pick]["coil_type"] = FIFF.FIFFV_COIL_ARTEMIS123_GRAD
    for info, n, name in [
        (read_info(fname_raw), 304, "306m"),
        (read_info(fname_kit_raw), 150, "KIT"),  # Delaunay
        (read_info(fname_bti_raw), 304, "Magnes"),
        (read_info(fname_ctf_raw), 342, "CTF"),
        (new_info, 102, "unknown"),  # Delaunay
        (artemis_info, 102, "ARTEMIS123"),  # Delaunay
    ]:
        with catch_logging() as log:
            helmet = get_meg_helmet_surf(info, trans, verbose=True)
        log = log.getvalue()
        assert name in log
        assert_equal(len(helmet["rr"]), n)
        assert_equal(len(helmet["rr"]), len(helmet["nn"]))


@testing.requires_testing_data
def test_head():
    """Test loading the head surface."""
    surf_1 = get_head_surf("sample", subjects_dir=subjects_dir)
    surf_2 = get_head_surf("sample", "head", subjects_dir=subjects_dir)
    assert len(surf_1["rr"]) < len(surf_2["rr"])  # BEM vs dense head
    pytest.raises(TypeError, get_head_surf, subject=None, subjects_dir=subjects_dir)


def test_fast_cross_3d():
    """Test cross product with lots of elements."""
    x = rng.rand(100000, 3)
    y = rng.rand(1, 3)
    z = np.cross(x, y)
    zz = fast_cross_3d(x, y)
    assert_array_equal(z, zz)
    # broadcasting and non-2D
    zz = fast_cross_3d(x[:, np.newaxis], y[0])
    assert_array_equal(z, zz[:, 0])


def test_compute_nearest():
    """Test nearest neighbor searches."""
    x = rng.randn(500, 3)
    x /= np.sqrt(np.sum(x**2, axis=1))[:, None]
    nn_true = rng.permutation(np.arange(500, dtype=np.int64))[:20]
    y = x[nn_true]

    nn1 = _compute_nearest(x, y, method="BallTree")
    nn2 = _compute_nearest(x, y, method="KDTree")
    nn3 = _compute_nearest(x, y, method="cdist")
    assert_array_equal(nn_true, nn1)
    assert_array_equal(nn_true, nn2)
    assert_array_equal(nn_true, nn3)

    # test distance support
    nnn1 = _compute_nearest(x, y, method="BallTree", return_dists=True)
    nnn2 = _compute_nearest(x, y, method="KDTree", return_dists=True)
    nnn3 = _compute_nearest(x, y, method="cdist", return_dists=True)
    assert_array_equal(nnn1[0], nn_true)
    assert_array_equal(nnn1[1], np.zeros_like(nn1))  # all dists should be 0
    assert_equal(len(nnn1), len(nnn2))
    for nn1, nn2, nn3 in zip(nnn1, nnn2, nnn3):
        assert_array_equal(nn1, nn2)
        assert_array_equal(nn1, nn3)


@testing.requires_testing_data
def test_io_surface(tmp_path):
    """Test reading and writing of Freesurfer surface mesh files."""
    pytest.importorskip("nibabel")
    fname_quad = data_path / "subjects" / "bert" / "surf" / "lh.inflated.nofix"
    fname_tri = data_path / "subjects" / "sample" / "bem" / "inner_skull.surf"
    for fname in (fname_quad, fname_tri):
        with _record_warnings():  # no volume info
            pts, tri, vol_info = read_surface(fname, read_metadata=True)
        write_surface(tmp_path / "tmp", pts, tri, volume_info=vol_info, overwrite=True)
        with _record_warnings():  # no volume info
            c_pts, c_tri, c_vol_info = read_surface(
                tmp_path / "tmp", read_metadata=True
            )
        assert_array_equal(pts, c_pts)
        assert_array_equal(tri, c_tri)
        assert_equal(object_diff(vol_info, c_vol_info), "")
        if fname != fname_tri:  # don't bother testing wavefront for the bigger
            continue

        # Test writing/reading a Wavefront .obj file
        write_surface(tmp_path / "tmp.obj", pts, tri, volume_info=None, overwrite=True)
        c_pts, c_tri = read_surface(tmp_path / "tmp.obj", read_metadata=False)
        assert_array_equal(pts, c_pts)
        assert_array_equal(tri, c_tri)

    # reading patches (just a smoke test, let the flatmap viz tests be more
    # complete)
    fname_patch = data_path / "subjects" / "fsaverage" / "surf" / "rh.cortex.patch.flat"
    _read_patch(fname_patch)


@testing.requires_testing_data
def test_read_curv():
    """Test reading curvature data."""
    pytest.importorskip("nibabel")
    fname_curv = data_path / "subjects" / "fsaverage" / "surf" / "lh.curv"
    fname_surf = data_path / "subjects" / "fsaverage" / "surf" / "lh.inflated"
    bin_curv = read_curvature(fname_curv)
    rr = read_surface(fname_surf)[0]
    assert len(bin_curv) == len(rr)
    assert np.logical_or(bin_curv == 0, bin_curv == 1).all()


@pytest.mark.parametrize("n_tri", (4, 3, 2))
def test_decimate_surface_vtk(n_tri):
    """Test triangular surface decimation."""
    pytest.importorskip("pyvista")
    points = np.array(
        [
            [-0.00686118, -0.10369860, 0.02615170],
            [-0.00713948, -0.10370162, 0.02614874],
            [-0.00686208, -0.10368247, 0.02588313],
            [-0.00713987, -0.10368724, 0.02587745],
        ]
    )
    tris = np.array([[0, 1, 2], [1, 2, 3], [0, 3, 1], [1, 2, 0]])
    _, this_tris = decimate_surface(points, tris, n_tri)
    want = (n_tri, n_tri - 1)
    if n_tri == 3:
        want = want + (1,)
    assert len(this_tris) in want
    with pytest.raises(ValueError, match="exceeds number of original"):
        decimate_surface(points, tris, len(tris) + 1)
    nirvana = 5
    tris = np.array([[0, 1, 2], [1, 2, 3], [0, 3, 1], [1, 2, nirvana]])
    with pytest.raises(ValueError, match="undefined points"):
        decimate_surface(points, tris, n_tri)


@requires_freesurfer("mris_sphere")
def test_decimate_surface_sphere():
    """Test sphere mode of decimation."""
    pytest.importorskip("nibabel")
    rr, tris = _tessellate_sphere(3)
    assert len(rr) == 66
    assert len(tris) == 128
    for kind, n_tri in [("ico", 20), ("oct", 32)]:
        with catch_logging() as log:
            _, tris_new = decimate_surface(
                rr, tris, n_tri, method="sphere", verbose=True
            )
        log = log.getvalue()
        assert "Freesurfer" in log
        assert kind in log
        assert len(tris_new) == n_tri


@pytest.mark.parametrize(
    "dig_kinds, exclude, count, bounds, outliers",
    [
        ("auto", False, 72, (0.001, 0.002), 0),
        (("eeg", "extra", "cardinal", "hpi"), False, 146, (0.002, 0.003), 1),
        (("eeg", "extra", "cardinal", "hpi"), True, 139, (0.001, 0.002), 0),
    ],
)
@testing.requires_testing_data
def test_dig_mri_distances(dig_kinds, exclude, count, bounds, outliers):
    """Test the trans obtained by coregistration."""
    info = read_info(fname_raw)
    dists = dig_mri_distances(
        info,
        fname_trans,
        "sample",
        subjects_dir,
        dig_kinds=dig_kinds,
        exclude_frontal=exclude,
    )
    assert dists.shape == (count,)
    assert bounds[0] < np.mean(dists) < bounds[1]
    assert np.sum(dists > 0.03) == outliers


def test_normal_orth():
    """Test _normal_orth."""
    nns = np.eye(3)
    for nn in nns:
        ori = _normal_orth(nn)
        assert_allclose(ori[2], nn, atol=1e-12)


# 0.06 s locally even with all these params
@pytest.mark.parametrize("dtype", (np.float64, np.uint16, ">i4"))
@pytest.mark.parametrize("order", "FC")
@pytest.mark.parametrize("value", (1, 12))
@pytest.mark.parametrize("smooth", (0, 0.9))
def test_marching_cubes(dtype, value, smooth, order):
    """Test creating surfaces via marching cubes."""
    pytest.importorskip("pyvista")
    data = np.zeros((50, 50, 50), dtype=dtype, order=order)
    data[20:30, 20:30, 20:30] = value
    level = [value]
    out = _marching_cubes(data, level, smooth=smooth)
    assert len(out) == 1
    verts, triangles = out[0]
    # verts and faces are rather large so use checksum
    rtol = 1e-2 if smooth else 1e-9
    assert_allclose(verts.sum(axis=0), [14700, 14700, 14700], rtol=rtol)
    tri_sum = triangles.sum(axis=0).tolist()
    assert tri_sum in ([350588, 360865, 363402], [350408, 359867, 364089])
    # test fill holes
    data[24:27, 24:27, 24:27] = 0
    verts, triangles = _marching_cubes(data, level, smooth=smooth, fill_hole_size=2)[0]
    # check that no surfaces in the middle
    assert np.linalg.norm(verts - np.array([25, 25, 25]), axis=1).min() > 4
    # problematic values
    with pytest.raises(TypeError, match="1D array-like"):
        _marching_cubes(data, ["foo"])
    with pytest.raises(TypeError, match="1D array-like"):
        _marching_cubes(data, [[1]])
    with pytest.raises(TypeError, match="1D array-like"):
        _marching_cubes(data, [1.0])
    with pytest.raises(ValueError, match="must be between 0"):
        _marching_cubes(data, [1], smooth=1.0)
    with pytest.raises(ValueError, match="3D data"):
        _marching_cubes(data[0], [1])


@testing.requires_testing_data
def test_get_montage_volume_labels():
    """Test finding ROI labels near montage channel locations."""
    pytest.importorskip("nibabel")
    ch_coords = np.array(
        [
            [-8.7040273, 17.99938754, 10.29604017],
            [-14.03007764, 19.69978401, 12.07236939],
            [-21.1130506, 21.98310911, 13.25658887],
        ]
    )
    ch_pos = dict(zip(["1", "2", "3"], ch_coords / 1000))  # mm -> m
    montage = make_dig_montage(ch_pos, coord_frame="mri")
    labels, colors = get_montage_volume_labels(
        montage, "sample", subjects_dir, aseg="aseg", dist=1
    )
    assert labels == {
        "1": ["Unknown"],
        "2": ["Left-Cerebral-Cortex"],
        "3": ["Left-Cerebral-Cortex"],
    }
    assert "Unknown" in colors
    assert "Left-Cerebral-Cortex" in colors
    np.testing.assert_almost_equal(
        colors["Left-Cerebral-Cortex"],
        (0.803921568627451, 0.24313725490196078, 0.3058823529411765, 1.0),
    )
    np.testing.assert_almost_equal(colors["Unknown"], (0.0, 0.0, 0.0, 1.0))

    # test inputs
    fail_montage = make_dig_montage(ch_pos, coord_frame="head")
    with pytest.raises(RuntimeError, match="Coordinate frame not supported"):
        get_montage_volume_labels(fail_montage, "sample", subjects_dir, aseg="aseg")
    with pytest.raises(ValueError, match="between 0 and 10"):
        get_montage_volume_labels(montage, "sample", subjects_dir, dist=11)


def test_voxel_neighbors():
    """Test finding points above a threshold near a seed location."""
    locs = np.array(np.meshgrid(*[np.linspace(-1, 1, 101)] * 3))
    image = 1 - np.linalg.norm(locs, axis=0)
    true_volume = set([tuple(coord) for coord in np.array(np.where(image > 0.95)).T])
    volume = _voxel_neighbors(
        np.array([-0.3, 0.6, 0.5]) + (np.array(image.shape[0]) - 1) / 2,
        image,
        thresh=0.95,
        use_relative=False,
    )
    assert volume.difference(true_volume) == set()
    assert true_volume.difference(volume) == set()


@testing.requires_testing_data
@pytest.mark.parametrize("ret_nn", (False, True))
@pytest.mark.parametrize("method", ("accurate", "nearest"))
def test_project_onto_surface(method, ret_nn):
    """Test _project_onto_surface (gh-10930)."""
    locs = np.random.default_rng(0).normal(size=(10, 3))
    locs *= 2 / np.linalg.norm(locs, axis=1)[:, None]  # lie on a sphere rad=2
    surf = _get_ico_surface(3)
    assert len(surf["rr"]) == 642
    assert_allclose(np.linalg.norm(surf["rr"], axis=1), 1.0, rtol=1e-3)  # unit
    # project
    weights, tri_idx, *out = _project_onto_surface(
        locs, surf, project_rrs=True, return_nn=ret_nn, method=method
    )
    locs /= 2.0  # back to unit
    assert_allclose(np.linalg.norm(locs, axis=1), 1.0, rtol=1e-5)
    assert len(out) == 2 if ret_nn else 1
    # for a sphere, both the rr (out[0]) and nn (out[1], if exists) should
    # both be very similar to each other and to our unit-length `locs`
    for kind, comp in zip(("rr", "nn"), out):
        assert_allclose(
            np.linalg.norm(comp, axis=1),
            1.0,
            atol=0.05,
            err_msg=f"{kind} not unit vectors for {method}",
        )
        cos = np.sum(locs * comp, axis=1)
        assert_allclose(
            cos,
            1.0,
            atol=0.05,  # ico > 3 would be even better tol
            err_msg=f"{kind} not in same direction as locs for {method}",
        )
