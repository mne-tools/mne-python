import os
import os.path as op
import warnings

import numpy as np
from nose.tools import assert_true, assert_raises
from numpy.testing import assert_array_equal, assert_equal, assert_allclose

from mne.datasets import testing
from mne import read_trans, write_trans
from mne.io import read_info
from mne.utils import _TempDir, run_tests_if_main
from mne.tests.common import assert_naming
from mne.transforms import (invert_transform, _get_trans,
                            rotation, rotation3d, rotation_angles, _find_trans,
                            combine_transforms, apply_trans, translation,
                            get_ras_to_neuromag_trans, _pol_to_cart,
                            quat_to_rot, rot_to_quat, _angle_between_quats,
                            _find_vector_rotation, _sph_to_cart, _cart_to_sph,
                            _topo_to_sph,
                            _SphericalSurfaceWarp as SphericalSurfaceWarp,
                            rotation3d_align_z_axis)

warnings.simplefilter('always')  # enable b/c these tests throw warnings

data_path = testing.data_path(download=False)
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc-trans.fif')
fname_eve = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc_raw-eve.fif')

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
fname_trans = op.join(base_dir, 'sample-audvis-raw-trans.txt')
test_fif_fname = op.join(base_dir, 'test_raw.fif')
ctf_fname = op.join(base_dir, 'test_ctf_raw.fif')
hp_fif_fname = op.join(base_dir, 'test_chpi_raw_sss.fif')


def test_tps():
    """Test TPS warping."""
    az = np.linspace(0., 2 * np.pi, 20, endpoint=False)
    pol = np.linspace(0, np.pi, 12)[1:-1]
    sph = np.array(np.meshgrid(1, az, pol, indexing='ij'))
    sph.shape = (3, -1)
    assert_equal(sph.shape[1], 200)
    source = _sph_to_cart(sph.T)
    destination = source.copy()
    destination *= 2
    destination[:, 0] += 1
    # fit with 100 points
    warp = SphericalSurfaceWarp()
    assert_true('no ' in repr(warp))
    warp.fit(source[::3], destination[::2])
    assert_true('oct5' in repr(warp))
    destination_est = warp.transform(source)
    assert_allclose(destination_est, destination, atol=1e-3)


@testing.requires_testing_data
def test_get_trans():
    """Test converting '-trans.txt' to '-trans.fif'"""
    trans = read_trans(fname)
    trans = invert_transform(trans)  # starts out as head->MRI, so invert
    trans_2 = _get_trans(fname_trans)[0]
    assert_equal(trans['from'], trans_2['from'])
    assert_equal(trans['to'], trans_2['to'])
    assert_allclose(trans['trans'], trans_2['trans'], rtol=1e-5, atol=1e-5)


@testing.requires_testing_data
def test_io_trans():
    """Test reading and writing of trans files."""
    tempdir = _TempDir()
    os.mkdir(op.join(tempdir, 'sample'))
    assert_raises(RuntimeError, _find_trans, 'sample', subjects_dir=tempdir)
    trans0 = read_trans(fname)
    fname1 = op.join(tempdir, 'sample', 'test-trans.fif')
    trans0.save(fname1)
    assert_true(fname1 == _find_trans('sample', subjects_dir=tempdir))
    trans1 = read_trans(fname1)

    # check all properties
    assert_true(trans0['from'] == trans1['from'])
    assert_true(trans0['to'] == trans1['to'])
    assert_array_equal(trans0['trans'], trans1['trans'])

    # check reading non -trans.fif files
    assert_raises(IOError, read_trans, fname_eve)

    # check warning on bad filenames
    with warnings.catch_warnings(record=True) as w:
        fname2 = op.join(tempdir, 'trans-test-bad-name.fif')
        write_trans(fname2, trans0)
    assert_naming(w, 'test_transforms.py', 1)


def test_get_ras_to_neuromag_trans():
    """Test the coordinate transformation from ras to neuromag."""
    # create model points in neuromag-like space
    rng = np.random.RandomState(0)
    anterior = [0, 1, 0]
    left = [-1, 0, 0]
    right = [.8, 0, 0]
    up = [0, 0, 1]
    rand_pts = rng.uniform(-1, 1, (3, 3))
    pts = np.vstack((anterior, left, right, up, rand_pts))

    # change coord system
    rx, ry, rz, tx, ty, tz = rng.uniform(-2 * np.pi, 2 * np.pi, 6)
    trans = np.dot(translation(tx, ty, tz), rotation(rx, ry, rz))
    pts_changed = apply_trans(trans, pts)

    # transform back into original space
    nas, lpa, rpa = pts_changed[:3]
    hsp_trans = get_ras_to_neuromag_trans(nas, lpa, rpa)
    pts_restored = apply_trans(hsp_trans, pts_changed)

    err = "Neuromag transformation failed"
    assert_allclose(pts_restored, pts, atol=1e-6, err_msg=err)


def _cartesian_to_sphere(x, y, z):
    """Convert using old function."""
    hypotxy = np.hypot(x, y)
    r = np.hypot(hypotxy, z)
    elev = np.arctan2(z, hypotxy)
    az = np.arctan2(y, x)
    return az, elev, r


def _sphere_to_cartesian(theta, phi, r):
    """Convert using old function."""
    z = r * np.sin(phi)
    rcos_phi = r * np.cos(phi)
    x = rcos_phi * np.cos(theta)
    y = rcos_phi * np.sin(theta)
    return x, y, z


def test_sph_to_cart():
    """Test conversion between sphere and cartesian."""
    # Simple test, expected value (11, 0, 0)
    r, theta, phi = 11., 0., np.pi / 2.
    z = r * np.cos(phi)
    rsin_phi = r * np.sin(phi)
    x = rsin_phi * np.cos(theta)
    y = rsin_phi * np.sin(theta)
    coord = _sph_to_cart(np.array([[r, theta, phi]]))[0]
    assert_allclose(coord, (x, y, z), atol=1e-7)
    assert_allclose(coord, (r, 0, 0), atol=1e-7)
    rng = np.random.RandomState(0)
    # round-trip test
    coords = rng.randn(10, 3)
    assert_allclose(_sph_to_cart(_cart_to_sph(coords)), coords, atol=1e-5)
    # equivalence tests to old versions
    for coord in coords:
        sph = _cart_to_sph(coord[np.newaxis])
        cart = _sph_to_cart(sph)
        sph_old = np.array(_cartesian_to_sphere(*coord))
        cart_old = _sphere_to_cartesian(*sph_old)
        sph_old[1] = np.pi / 2. - sph_old[1]  # new convention
        assert_allclose(sph[0], sph_old[[2, 0, 1]], atol=1e-7)
        assert_allclose(cart[0], cart_old, atol=1e-7)
        assert_allclose(cart[0], coord, atol=1e-7)


def _polar_to_cartesian(theta, r):
    """Transform polar coordinates to cartesian"""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def test_polar_to_cartesian():
    """Test helper transform function from polar to cartesian"""
    r = 1
    theta = np.pi
    # expected values are (-1, 0)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    coord = _pol_to_cart(np.array([[r, theta]]))[0]
    # np.pi is an approx since pi is irrational
    assert_allclose(coord, (x, y), atol=1e-7)
    assert_allclose(coord, (-1, 0), atol=1e-7)
    assert_allclose(coord, _polar_to_cartesian(theta, r), atol=1e-7)
    rng = np.random.RandomState(0)
    r = rng.randn(10)
    theta = rng.rand(10) * (2 * np.pi)
    polar = np.array((r, theta)).T
    assert_allclose([_polar_to_cartesian(p[1], p[0]) for p in polar],
                    _pol_to_cart(polar), atol=1e-7)


def _topo_to_sphere(theta, radius):
    """Convert using old function."""
    sph_phi = (0.5 - radius) * 180
    sph_theta = -theta
    return sph_phi, sph_theta


def test_topo_to_sph():
    """Test topo to sphere conversion."""
    rng = np.random.RandomState(0)
    angles = rng.rand(10) * 360
    radii = rng.rand(10)
    angles[0] = 30
    radii[0] = 0.25
    # new way
    sph = _topo_to_sph(np.array([angles, radii]).T)
    new = _sph_to_cart(sph)
    new[:, [0, 1]] = new[:, [1, 0]] * [-1, 1]
    # old way
    for ii, (angle, radius) in enumerate(zip(angles, radii)):
        sph_phi, sph_theta = _topo_to_sphere(angle, radius)
        if ii == 0:
            assert_allclose(_topo_to_sphere(angle, radius), [45, -30])
        azimuth = sph_theta / 180.0 * np.pi
        elevation = sph_phi / 180.0 * np.pi
        assert_allclose(sph[ii], [1., azimuth, np.pi / 2. - elevation],
                        atol=1e-7)
        r = np.ones_like(radius)
        x, y, z = _sphere_to_cartesian(azimuth, elevation, r)
        pos = [-y, x, z]
        if ii == 0:
            expected = np.array([1. / 2., np.sqrt(3) / 2., 1.])
            expected /= np.sqrt(2)
            assert_allclose(pos, expected, atol=1e-7)
        assert_allclose(pos, new[ii], atol=1e-7)


def test_rotation():
    """Test conversion between rotation angles and transformation matrix."""
    tests = [(0, 0, 1), (.5, .5, .5), (np.pi, 0, -1.5)]
    for rot in tests:
        x, y, z = rot
        m = rotation3d(x, y, z)
        m4 = rotation(x, y, z)
        assert_array_equal(m, m4[:3, :3])
        back = rotation_angles(m)
        assert_equal(back, rot)
        back4 = rotation_angles(m4)
        assert_equal(back4, rot)


def test_rotation3d_align_z_axis():
    """Test rotation3d_align_z_axis."""
    # The more complex z axis fails the assert presumably due to tolerance
    #
    inp_zs = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, -1],
              [-0.75071668, -0.62183808,  0.22302888]]

    exp_res = [[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
               [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]],
               [[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]],
               [[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]],
               [[0.53919688, -0.38169517, -0.75071668],
                [-0.38169517, 0.683832, -0.62183808],
                [0.75071668, 0.62183808, 0.22302888]]]

    for res, z in zip(exp_res, inp_zs):
        assert_allclose(res, rotation3d_align_z_axis(z), atol=1e-7)


@testing.requires_testing_data
def test_combine():
    """Test combining transforms
    """
    trans = read_trans(fname)
    inv = invert_transform(trans)
    combine_transforms(trans, inv, trans['from'], trans['from'])
    assert_raises(RuntimeError, combine_transforms, trans, inv,
                  trans['to'], trans['from'])
    assert_raises(RuntimeError, combine_transforms, trans, inv,
                  trans['from'], trans['to'])
    assert_raises(RuntimeError, combine_transforms, trans, trans,
                  trans['from'], trans['to'])


def test_quaternions():
    """Test quaternion calculations."""
    rots = [np.eye(3)]
    for fname in [test_fif_fname, ctf_fname, hp_fif_fname]:
        rots += [read_info(fname)['dev_head_t']['trans'][:3, :3]]
    # nasty numerical cases
    rots += [np.array([
        [-0.99978541, -0.01873462, -0.00898756],
        [-0.01873462, 0.62565561, 0.77987608],
        [-0.00898756, 0.77987608, -0.62587152],
    ])]
    rots += [np.array([
        [0.62565561, -0.01873462, 0.77987608],
        [-0.01873462, -0.99978541, -0.00898756],
        [0.77987608, -0.00898756, -0.62587152],
    ])]
    rots += [np.array([
        [-0.99978541, -0.00898756, -0.01873462],
        [-0.00898756, -0.62587152, 0.77987608],
        [-0.01873462, 0.77987608, 0.62565561],
    ])]
    for rot in rots:
        assert_allclose(rot, quat_to_rot(rot_to_quat(rot)),
                        rtol=1e-5, atol=1e-5)
        rot = rot[np.newaxis, np.newaxis, :, :]
        assert_allclose(rot, quat_to_rot(rot_to_quat(rot)),
                        rtol=1e-5, atol=1e-5)

    # let's make sure our angle function works in some reasonable way
    for ii in range(3):
        for jj in range(3):
            a = np.zeros(3)
            b = np.zeros(3)
            a[ii] = 1.
            b[jj] = 1.
            expected = np.pi if ii != jj else 0.
            assert_allclose(_angle_between_quats(a, b), expected, atol=1e-5)


def test_vector_rotation():
    """Test basic rotation matrix math."""
    x = np.array([1., 0., 0.])
    y = np.array([0., 1., 0.])
    rot = _find_vector_rotation(x, y)
    assert_array_equal(rot,
                       [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    quat_1 = rot_to_quat(rot)
    quat_2 = rot_to_quat(np.eye(3))
    assert_allclose(_angle_between_quats(quat_1, quat_2), np.pi / 2.)

run_tests_if_main()
