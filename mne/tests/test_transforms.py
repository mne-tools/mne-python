import os
import os.path as op
import numpy as np

from nose.tools import assert_true, assert_raises
from numpy.testing import (assert_array_equal, assert_equal, assert_allclose,
                           assert_almost_equal, assert_array_almost_equal)
import warnings

from mne.datasets import testing
from mne import read_trans, write_trans
from mne.io import read_info
from mne.utils import _TempDir, run_tests_if_main
from mne.tests.common import assert_naming
from mne.transforms import (invert_transform, _get_trans,
                            rotation, rotation3d, rotation_angles, _find_trans,
                            combine_transforms, apply_trans, translation,
                            get_ras_to_neuromag_trans, _sphere_to_cartesian,
                            _polar_to_cartesian, _cartesian_to_sphere,
                            quat_to_rot, rot_to_quat, _angle_between_quats,
                            _find_vector_rotation)

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
    """Test reading and writing of trans files
    """
    tempdir = _TempDir()
    os.mkdir(op.join(tempdir, 'sample'))
    assert_raises(RuntimeError, _find_trans, 'sample', subjects_dir=tempdir)
    trans0 = read_trans(fname)
    fname1 = op.join(tempdir, 'sample', 'test-trans.fif')
    write_trans(fname1, trans0)
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
    """Test the coordinate transformation from ras to neuromag"""
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
    assert_array_almost_equal(pts_restored, pts, 6, err)


def test_sphere_to_cartesian():
    """Test helper transform function from sphere to cartesian"""
    phi, theta, r = (np.pi, np.pi, 1)
    # expected value is (1, 0, 0)
    z = r * np.sin(phi)
    rcos_phi = r * np.cos(phi)
    x = rcos_phi * np.cos(theta)
    y = rcos_phi * np.sin(theta)
    coord = _sphere_to_cartesian(phi, theta, r)
    # np.pi is an approx since pi is irrational
    assert_almost_equal(coord, (x, y, z), 10)
    assert_almost_equal(coord, (1, 0, 0), 10)


def test_polar_to_cartesian():
    """Test helper transform function from polar to cartesian"""
    r = 1
    theta = np.pi
    # expected values are (-1, 0)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    coord = _polar_to_cartesian(theta, r)
    # np.pi is an approx since pi is irrational
    assert_almost_equal(coord, (x, y), 10)
    assert_almost_equal(coord, (-1, 0), 10)


def test_cartesian_to_sphere():
    """Test helper transform function from cartesian to sphere"""
    x, y, z = (1, 0, 0)
    # expected values are (0, 0, 1)
    hypotxy = np.hypot(x, y)
    r = np.hypot(hypotxy, z)
    elev = np.arctan2(z, hypotxy)
    az = np.arctan2(y, x)
    coord = _cartesian_to_sphere(x, y, z)
    assert_equal(coord, (az, elev, r))
    assert_equal(coord, (0, 0, 1))


def test_rotation():
    """Test conversion between rotation angles and transformation matrix
    """
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
    """Test quaternion calculations
    """
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
    """Test basic rotation matrix math
    """
    x = np.array([1., 0., 0.])
    y = np.array([0., 1., 0.])
    rot = _find_vector_rotation(x, y)
    assert_array_equal(rot,
                       [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    quat_1 = rot_to_quat(rot)
    quat_2 = rot_to_quat(np.eye(3))
    assert_allclose(_angle_between_quats(quat_1, quat_2), np.pi / 2.)

run_tests_if_main()
