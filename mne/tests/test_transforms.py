import os
import os.path as op

import pytest
import numpy as np
from numpy.testing import (assert_array_equal, assert_equal, assert_allclose,
                           assert_array_less, assert_almost_equal)
import itertools

import mne
from mne.datasets import testing
from mne.fixes import _get_img_fdata
from mne import read_trans, write_trans
from mne.io import read_info
from mne.transforms import (invert_transform, _get_trans,
                            rotation, rotation3d, rotation_angles, _find_trans,
                            combine_transforms, apply_trans, translation,
                            get_ras_to_neuromag_trans, _pol_to_cart,
                            quat_to_rot, rot_to_quat, _angle_between_quats,
                            _find_vector_rotation, _sph_to_cart, _cart_to_sph,
                            _topo_to_sph, _average_quats,
                            _SphericalSurfaceWarp as SphericalSurfaceWarp,
                            rotation3d_align_z_axis, _read_fs_xfm,
                            _write_fs_xfm, _quat_real, _fit_matched_points,
                            _quat_to_euler, _euler_to_quat,
                            _quat_to_affine, _compute_r2, _validate_pipeline)
from mne.utils import requires_nibabel, requires_dipy

data_path = testing.data_path(download=False)
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc-trans.fif')
fname_eve = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc_raw-eve.fif')
subjects_dir = op.join(data_path, 'subjects')
fname_t1 = op.join(subjects_dir, 'fsaverage', 'mri', 'T1.mgz')

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
    assert 'no ' in repr(warp)
    warp.fit(source[::3], destination[::2])
    assert 'oct5' in repr(warp)
    destination_est = warp.transform(source)
    assert_allclose(destination_est, destination, atol=1e-3)


@testing.requires_testing_data
def test_get_trans():
    """Test converting '-trans.txt' to '-trans.fif'."""
    trans = read_trans(fname)
    trans = invert_transform(trans)  # starts out as head->MRI, so invert
    trans_2 = _get_trans(fname_trans)[0]
    assert trans.__eq__(trans_2, atol=1e-5)


@testing.requires_testing_data
def test_io_trans(tmp_path):
    """Test reading and writing of trans files."""
    tempdir = str(tmp_path)
    os.mkdir(op.join(tempdir, 'sample'))
    pytest.raises(RuntimeError, _find_trans, 'sample', subjects_dir=tempdir)
    trans0 = read_trans(fname)
    fname1 = op.join(tempdir, 'sample', 'test-trans.fif')
    trans0.save(fname1)
    assert fname1 == _find_trans('sample', subjects_dir=tempdir)
    trans1 = read_trans(fname1)

    # check all properties
    assert trans0 == trans1

    # check reading non -trans.fif files
    pytest.raises(IOError, read_trans, fname_eve)

    # check warning on bad filenames
    fname2 = op.join(tempdir, 'trans-test-bad-name.fif')
    with pytest.warns(RuntimeWarning, match='-trans.fif'):
        write_trans(fname2, trans0)


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
    """Transform polar coordinates to cartesian."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def test_polar_to_cartesian():
    """Test helper transform function from polar to cartesian."""
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


def _topo_to_phi_theta(theta, radius):
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
        sph_phi, sph_theta = _topo_to_phi_theta(angle, radius)
        if ii == 0:
            assert_allclose(_topo_to_phi_theta(angle, radius), [45, -30])
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
        assert_almost_equal(actual=back, desired=rot, decimal=12)
        back4 = rotation_angles(m4)
        assert_almost_equal(actual=back4, desired=rot, decimal=12)


def test_rotation3d_align_z_axis():
    """Test rotation3d_align_z_axis."""
    # The more complex z axis fails the assert presumably due to tolerance
    #
    inp_zs = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, -1],
              [-0.75071668, -0.62183808, 0.22302888]]

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
    """Test combining transforms."""
    trans = read_trans(fname)
    inv = invert_transform(trans)
    combine_transforms(trans, inv, trans['from'], trans['from'])
    pytest.raises(RuntimeError, combine_transforms, trans, inv,
                  trans['to'], trans['from'])
    pytest.raises(RuntimeError, combine_transforms, trans, inv,
                  trans['from'], trans['to'])
    pytest.raises(RuntimeError, combine_transforms, trans, trans,
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

    y_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1.]])
    assert_allclose(_angle_between_quats(rot_to_quat(y_180),
                                         np.zeros(3)), np.pi)
    h_180_attitude_90 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.]])
    assert_allclose(_angle_between_quats(rot_to_quat(h_180_attitude_90),
                                         np.zeros(3)), np.pi)


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


def test_average_quats():
    """Test averaging of quaternions."""
    sq2 = 1. / np.sqrt(2.)
    quats = np.array([[0, sq2, sq2],
                      [0, sq2, sq2],
                      [0, sq2, 0],
                      [0, 0, sq2],
                      [sq2, 0, 0]], float)
    # In MATLAB:
    # quats = [[0, sq2, sq2, 0]; [0, sq2, sq2, 0];
    #          [0, sq2, 0, sq2]; [0, 0, sq2, sq2]; [sq2, 0, 0, sq2]];
    expected = [quats[0],
                quats[0],
                [0, 0.788675134594813, 0.577350269189626],
                [0, 0.657192299694123, 0.657192299694123],
                [0.100406058540540, 0.616329446922803, 0.616329446922803]]
    # Averaging the first two should give the same thing:
    for lim, ex in enumerate(expected):
        assert_allclose(_average_quats(quats[:lim + 1]), ex, atol=1e-7)
    quats[1] *= -1  # same quaternion (hidden value is zero here)!
    rot_0, rot_1 = quat_to_rot(quats[:2])
    assert_allclose(rot_0, rot_1, atol=1e-7)
    for lim, ex in enumerate(expected):
        assert_allclose(_average_quats(quats[:lim + 1]), ex, atol=1e-7)
    # Assert some symmetry
    count = 0
    extras = [[sq2, sq2, 0]] + list(np.eye(3))
    for quat in np.concatenate((quats, expected, extras)):
        if np.isclose(_quat_real(quat), 0., atol=1e-7):  # can flip sign
            count += 1
            angle = _angle_between_quats(quat, -quat)
            assert_allclose(angle, 0., atol=1e-7)
            rot_0, rot_1 = quat_to_rot(np.array((quat, -quat)))
            assert_allclose(rot_0, rot_1, atol=1e-7)
    assert count == 4 + len(extras)


@testing.requires_testing_data
@pytest.mark.parametrize('subject', ('fsaverage', 'sample'))
def test_fs_xfm(subject, tmp_path):
    """Test reading and writing of Freesurfer transforms."""
    fname = op.join(data_path, 'subjects', subject, 'mri', 'transforms',
                    'talairach.xfm')
    xfm, kind = _read_fs_xfm(fname)
    if subject == 'fsaverage':
        assert_allclose(xfm, np.eye(4), atol=1e-5)  # fsaverage is in MNI
    assert kind == 'MNI Transform File'
    tempdir = str(tmp_path)
    fname_out = op.join(tempdir, 'out.xfm')
    _write_fs_xfm(fname_out, xfm, kind)
    xfm_read, kind_read = _read_fs_xfm(fname_out)
    assert kind_read == kind
    assert_allclose(xfm, xfm_read, rtol=1e-5, atol=1e-5)
    # Some wacky one
    xfm[:3] = np.random.RandomState(0).randn(3, 4)
    _write_fs_xfm(fname_out, xfm, 'foo')
    xfm_read, kind_read = _read_fs_xfm(fname_out)
    assert kind_read == 'foo'
    assert_allclose(xfm, xfm_read, rtol=1e-5, atol=1e-5)
    # degenerate conditions
    with open(fname_out, 'w') as fid:
        fid.write('foo')
    with pytest.raises(ValueError, match='Failed to find'):
        _read_fs_xfm(fname_out)
    _write_fs_xfm(fname_out, xfm[:2], 'foo')
    with pytest.raises(ValueError, match='Could not find'):
        _read_fs_xfm(fname_out)


@pytest.fixture()
def quats():
    """Make some unit quats."""
    quats = np.random.RandomState(0).randn(5, 3)
    quats[:, 0] = 0  # identity
    quats /= 2 * np.linalg.norm(quats, axis=1, keepdims=True)  # some real part
    return quats


def _check_fit_matched_points(
        p, x, weights, do_scale, angtol=1e-5, dtol=1e-5, stol=1e-7):
    __tracebackhide__ = True
    mne.coreg._ALLOW_ANALITICAL = False
    try:
        params = mne.coreg.fit_matched_points(
            p, x, weights=weights, scale=do_scale, out='params')
    finally:
        mne.coreg._ALLOW_ANALITICAL = True
    quat_an, scale_an = _fit_matched_points(p, x, weights, scale=do_scale)
    assert len(params) == 6 + int(do_scale)
    q_co = _euler_to_quat(params[:3])
    translate_co = params[3:6]
    angle = np.rad2deg(_angle_between_quats(quat_an[:3], q_co))
    dist = np.linalg.norm(quat_an[3:] - translate_co)
    assert 0 <= angle < angtol, 'angle'
    assert 0 <= dist < dtol, 'dist'
    if do_scale:
        scale_co = params[6]
        assert_allclose(scale_an, scale_co, rtol=stol, err_msg='scale')
    # errs
    trans = _quat_to_affine(quat_an)
    trans[:3, :3] *= scale_an
    weights = np.ones(1) if weights is None else weights
    err_an = np.linalg.norm(
        weights[:, np.newaxis] * apply_trans(trans, p) - x)
    trans = mne.coreg._trans_from_params((True, True, do_scale), params)
    err_co = np.linalg.norm(
        weights[:, np.newaxis] * apply_trans(trans, p) - x)
    if err_an > 1e-14:
        assert err_an < err_co * 1.5
    return quat_an, scale_an


@pytest.mark.parametrize('scaling', [0.25, 1])
@pytest.mark.parametrize('do_scale', (True, False))
def test_fit_matched_points(quats, scaling, do_scale):
    """Test analytical least-squares matched point fitting."""
    if scaling != 1 and not do_scale:
        return  # no need to test this, it will not be good
    rng = np.random.RandomState(0)
    fro = rng.randn(10, 3)
    translation = rng.randn(3)
    for qi, quat in enumerate(quats):
        to = scaling * np.dot(quat_to_rot(quat), fro.T).T + translation
        for corrupted in (False, True):
            # mess up a point
            if corrupted:
                to[0, 2] += 100
                weights = np.ones(len(to))
                weights[0] = 0
            else:
                weights = None
            est, scale_est = _check_fit_matched_points(
                fro, to, weights=weights, do_scale=do_scale)
            assert_allclose(scale_est, scaling, rtol=1e-5)
            assert_allclose(est[:3], quat, atol=1e-14)
            assert_allclose(est[3:], translation, atol=1e-14)
        # if we don't adjust for the corruption above, it should get worse
        angle = dist = None
        for weighted in (False, True):
            if not weighted:
                weights = None
                dist_bounds = (5, 20)
                if scaling == 1:
                    angle_bounds = (5, 95)
                    angtol, dtol, stol = 1, 15, 3
                else:
                    angle_bounds = (5, 105)
                    angtol, dtol, stol = 20, 15, 3
            else:
                weights = np.ones(len(to))
                weights[0] = 10  # weighted=True here means "make it worse"
                angle_bounds = (angle, 180)  # unweighted values as new min
                dist_bounds = (dist, 100)
                if scaling == 1:
                    # XXX this angtol is not great but there is a hard to
                    # identify linalg/angle calculation bug on Travis...
                    angtol, dtol, stol = 180, 70, 3
                else:
                    angtol, dtol, stol = 50, 70, 3
            est, scale_est = _check_fit_matched_points(
                fro, to, weights=weights, do_scale=do_scale,
                angtol=angtol, dtol=dtol, stol=stol)
            assert not np.allclose(est[:3], quat, atol=1e-5)
            assert not np.allclose(est[3:], translation, atol=1e-5)
            angle = np.rad2deg(_angle_between_quats(est[:3], quat))
            assert_array_less(angle_bounds[0], angle)
            assert_array_less(angle, angle_bounds[1])
            dist = np.linalg.norm(est[3:] - translation)
            assert_array_less(dist_bounds[0], dist)
            assert_array_less(dist, dist_bounds[1])


def test_euler(quats):
    """Test euler transformations."""
    euler = _quat_to_euler(quats)
    quats_2 = _euler_to_quat(euler)
    assert_allclose(quats, quats_2, atol=1e-14)
    quat_rot = quat_to_rot(quats)
    euler_rot = np.array([rotation(*e)[:3, :3] for e in euler])
    assert_allclose(quat_rot, euler_rot, atol=1e-14)


@requires_nibabel()
@requires_dipy()
@pytest.mark.slowtest
@testing.requires_testing_data
def test_volume_registration():
    """Test volume registration."""
    import nibabel as nib
    from dipy.align import resample
    T1 = nib.load(fname_t1)
    affine = np.eye(4)
    affine[0, 3] = 10
    T1_resampled = resample(moving=T1.get_fdata(),
                            static=T1.get_fdata(),
                            moving_affine=T1.affine,
                            static_affine=T1.affine,
                            between_affine=np.linalg.inv(affine))
    for pipeline in ('rigids', ('translation', 'sdr')):
        reg_affine, sdr_morph = mne.transforms.compute_volume_registration(
            T1_resampled, T1, pipeline=pipeline, zooms=10, niter=[5])
        assert_allclose(affine, reg_affine, atol=0.01)
        T1_aligned = mne.transforms.apply_volume_registration(
            T1_resampled, T1, reg_affine, sdr_morph)
        r2 = _compute_r2(_get_img_fdata(T1_aligned), _get_img_fdata(T1))
        assert 99.9 < r2

    # check that all orders of the pipeline work
    for pipeline_len in range(1, 5):
        for pipeline in itertools.combinations(
                ('translation', 'rigid', 'affine', 'sdr'), pipeline_len):
            _validate_pipeline(pipeline)
            _validate_pipeline(list(pipeline))

    with pytest.raises(ValueError, match='Steps in pipeline are out of order'):
        _validate_pipeline(('sdr', 'affine'))

    with pytest.raises(ValueError,
                       match='Steps in pipeline should not be repeated'):
        _validate_pipeline(('affine', 'affine'))
