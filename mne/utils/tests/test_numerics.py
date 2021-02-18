from copy import deepcopy
from io import StringIO
import os.path as op
from datetime import datetime, timezone

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pytest
from scipy import sparse

from mne import read_evokeds, read_cov, pick_types
from mne.io.pick import _picks_by_type
from mne.epochs import make_fixed_length_epochs
from mne.io import read_raw_fif
from mne.time_frequency import tfr_morlet
from mne.utils import (_get_inst_data, hashfunc,
                       sum_squared, compute_corr, create_slices, _time_mask,
                       _freq_mask, random_permutation, _reg_pinv, object_size,
                       object_hash, object_diff, _apply_scaling_cov,
                       _undo_scaling_cov, _apply_scaling_array,
                       _undo_scaling_array, _PCA, requires_sklearn,
                       _array_equal_nan, _julian_to_cal, _cal_to_julian,
                       _dt_to_julian, _julian_to_dt, grand_average,
                       _ReuseCycle, requires_version)


base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
fname_raw = op.join(base_dir, 'test_raw.fif')
ave_fname = op.join(base_dir, 'test-ave.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')


def test_get_inst_data():
    """Test _get_inst_data."""
    raw = read_raw_fif(fname_raw)
    raw.crop(tmax=1.)
    assert_array_equal(_get_inst_data(raw), raw._data)
    raw.pick_channels(raw.ch_names[:2])

    epochs = make_fixed_length_epochs(raw, 0.5)
    assert_array_equal(_get_inst_data(epochs), epochs._data)

    evoked = epochs.average()
    assert_array_equal(_get_inst_data(evoked), evoked.data)

    evoked.crop(tmax=0.1)
    picks = list(range(2))
    freqs = [50., 55.]
    n_cycles = 3
    tfr = tfr_morlet(evoked, freqs, n_cycles, return_itc=False, picks=picks)
    assert_array_equal(_get_inst_data(tfr), tfr.data)

    pytest.raises(TypeError, _get_inst_data, 'foo')


def test_hashfunc(tmpdir):
    """Test md5/sha1 hash calculations."""
    tempdir = str(tmpdir)
    fname1 = op.join(tempdir, 'foo')
    fname2 = op.join(tempdir, 'bar')
    with open(fname1, 'wb') as fid:
        fid.write(b'abcd')
    with open(fname2, 'wb') as fid:
        fid.write(b'efgh')

    for hash_type in ('md5', 'sha1'):
        hash1 = hashfunc(fname1, hash_type=hash_type)
        hash1_ = hashfunc(fname1, 1, hash_type=hash_type)

        hash2 = hashfunc(fname2, hash_type=hash_type)
        hash2_ = hashfunc(fname2, 1024, hash_type=hash_type)

        assert hash1 == hash1_
        assert hash2 == hash2_
        assert hash1 != hash2


def test_sum_squared():
    """Test optimized sum of squares."""
    X = np.random.RandomState(0).randint(0, 50, (3, 3))
    assert np.sum(X ** 2) == sum_squared(X)


def test_compute_corr():
    """Test Anscombe's Quartett."""
    x = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
    y = np.array([[8.04, 6.95, 7.58, 8.81, 8.33, 9.96,
                   7.24, 4.26, 10.84, 4.82, 5.68],
                  [9.14, 8.14, 8.74, 8.77, 9.26, 8.10,
                   6.13, 3.10, 9.13, 7.26, 4.74],
                  [7.46, 6.77, 12.74, 7.11, 7.81, 8.84,
                   6.08, 5.39, 8.15, 6.42, 5.73],
                  [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
                  [6.58, 5.76, 7.71, 8.84, 8.47, 7.04,
                   5.25, 12.50, 5.56, 7.91, 6.89]])

    r = compute_corr(x, y.T)
    r2 = np.array([np.corrcoef(x, y[i])[0, 1]
                   for i in range(len(y))])
    assert_allclose(r, r2)
    pytest.raises(ValueError, compute_corr, [1, 2], [])


def test_create_slices():
    """Test checking the create of time create_slices."""
    # Test that create_slices default provide an empty list
    assert (create_slices(0, 0) == [])
    # Test that create_slice return correct number of slices
    assert (len(create_slices(0, 100)) == 100)
    # Test with non-zero start parameters
    assert (len(create_slices(50, 100)) == 50)
    # Test slices' length with non-zero start and window_width=2
    assert (len(create_slices(0, 100, length=2)) == 50)
    # Test slices' length with manual slice separation
    assert (len(create_slices(0, 100, step=10)) == 10)
    # Test slices' within length for non-consecutive samples
    assert (len(create_slices(0, 500, length=50, step=10)) == 46)
    # Test that slices elements start, stop and step correctly
    slices = create_slices(0, 10)
    assert (slices[0].start == 0)
    assert (slices[0].step == 1)
    assert (slices[0].stop == 1)
    assert (slices[-1].stop == 10)
    # Same with larger window width
    slices = create_slices(0, 9, length=3)
    assert (slices[0].start == 0)
    assert (slices[0].step == 1)
    assert (slices[0].stop == 3)
    assert (slices[-1].stop == 9)
    # Same with manual slices' separation
    slices = create_slices(0, 9, length=3, step=1)
    assert (len(slices) == 7)
    assert (slices[0].step == 1)
    assert (slices[0].stop == 3)
    assert (slices[-1].start == 6)
    assert (slices[-1].stop == 9)


def test_time_mask():
    """Test safe time masking."""
    N = 10
    x = np.arange(N).astype(float)
    assert _time_mask(x, 0, N - 1).sum() == N
    assert _time_mask(x - 1e-10, 0, N - 1, sfreq=1000.).sum() == N
    assert _time_mask(x - 1e-10, None, N - 1, sfreq=1000.).sum() == N
    assert _time_mask(x - 1e-10, None, None, sfreq=1000.).sum() == N
    assert _time_mask(x - 1e-10, -np.inf, None, sfreq=1000.).sum() == N
    assert _time_mask(x - 1e-10, None, np.inf, sfreq=1000.).sum() == N
    # non-uniformly spaced inputs
    x = np.array([4, 10])
    assert _time_mask(x[:1], tmin=10, sfreq=1, raise_error=False).sum() == 0
    assert _time_mask(x[:1], tmin=11, tmax=12, sfreq=1,
                      raise_error=False).sum() == 0
    assert _time_mask(x, tmin=10, sfreq=1).sum() == 1
    assert _time_mask(x, tmin=6, sfreq=1).sum() == 1
    assert _time_mask(x, tmin=5, sfreq=1).sum() == 1
    assert _time_mask(x, tmin=4.5001, sfreq=1).sum() == 1
    assert _time_mask(x, tmin=4.4999, sfreq=1).sum() == 2
    assert _time_mask(x, tmin=4, sfreq=1).sum() == 2
    # degenerate cases
    with pytest.raises(ValueError, match='No samples remain'):
        _time_mask(x[:1], tmin=11, tmax=12)
    with pytest.raises(ValueError, match='must be less than or equal to tmax'):
        _time_mask(x[:1], tmin=10, sfreq=1)


def test_freq_mask():
    """Test safe frequency masking."""
    N = 10
    x = np.arange(N).astype(float)
    assert _freq_mask(x, 1000., fmin=0, fmax=N - 1).sum() == N
    assert _freq_mask(x - 1e-10, 1000., fmin=0, fmax=N - 1).sum() == N
    assert _freq_mask(x - 1e-10, 1000., fmin=None, fmax=N - 1).sum() == N
    assert _freq_mask(x - 1e-10, 1000., fmin=None, fmax=None).sum() == N
    assert _freq_mask(x - 1e-10, 1000., fmin=-np.inf, fmax=None).sum() == N
    assert _freq_mask(x - 1e-10, 1000., fmin=None, fmax=np.inf).sum() == N
    # non-uniformly spaced inputs
    x = np.array([4, 10])
    assert _freq_mask(x[:1], 1, fmin=10, raise_error=False).sum() == 0
    assert _freq_mask(x[:1], 1, fmin=11, fmax=12,
                      raise_error=False).sum() == 0
    assert _freq_mask(x, sfreq=1, fmin=10).sum() == 1
    assert _freq_mask(x, sfreq=1, fmin=6).sum() == 1
    assert _freq_mask(x, sfreq=1, fmin=5).sum() == 1
    assert _freq_mask(x, sfreq=1, fmin=4.5001).sum() == 1
    assert _freq_mask(x, sfreq=1, fmin=4.4999).sum() == 2
    assert _freq_mask(x, sfreq=1, fmin=4).sum() == 2
    # degenerate cases
    with pytest.raises(ValueError, match='sfreq can not be None'):
        _freq_mask(x[:1], sfreq=None, fmin=3, fmax=5)
    with pytest.raises(ValueError, match='No frequencies remain'):
        _freq_mask(x[:1], sfreq=1, fmin=11, fmax=12)
    with pytest.raises(ValueError, match='must be less than or equal to fmax'):
        _freq_mask(x[:1], sfreq=1, fmin=10)


def test_random_permutation():
    """Test random permutation function."""
    n_samples = 10
    random_state = 42
    python_randperm = random_permutation(n_samples, random_state)

    # matlab output when we execute rng(42), randperm(10)
    matlab_randperm = np.array([7, 6, 5, 1, 4, 9, 10, 3, 8, 2])

    assert_array_equal(python_randperm, matlab_randperm - 1)


def test_cov_scaling():
    """Test rescaling covs."""
    evoked = read_evokeds(ave_fname, condition=0, baseline=(None, 0),
                          proj=True)
    cov = read_cov(cov_fname)['data']
    cov2 = read_cov(cov_fname)['data']

    assert_array_equal(cov, cov2)
    evoked.pick_channels([evoked.ch_names[k] for k in pick_types(
        evoked.info, meg=True, eeg=True
    )])
    picks_list = _picks_by_type(evoked.info)
    scalings = dict(mag=1e15, grad=1e13, eeg=1e6)

    _apply_scaling_cov(cov2, picks_list, scalings=scalings)
    _apply_scaling_cov(cov, picks_list, scalings=scalings)
    assert_array_equal(cov, cov2)
    assert cov.max() > 1

    _undo_scaling_cov(cov2, picks_list, scalings=scalings)
    _undo_scaling_cov(cov, picks_list, scalings=scalings)
    assert_array_equal(cov, cov2)
    assert cov.max() < 1

    data = evoked.data.copy()
    _apply_scaling_array(data, picks_list, scalings=scalings)
    _undo_scaling_array(data, picks_list, scalings=scalings)
    assert_allclose(data, evoked.data, atol=1e-20)


@requires_version('numpy', '1.17')  # hermitian kwarg
@pytest.mark.parametrize('ndim', (2, 3))
def test_reg_pinv(ndim):
    """Test regularization and inversion of covariance matrix."""
    # create rank-deficient array
    a = np.array([[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]])
    for _ in range(ndim - 2):
        a = a[np.newaxis]

    # Test if rank-deficient matrix without regularization throws
    # specific warning
    with pytest.warns(RuntimeWarning, match='deficient'):
        _reg_pinv(a, reg=0.)

    # Test inversion with explicit rank
    a_inv_np = np.linalg.pinv(a, hermitian=True)
    a_inv_mne, loading_factor, rank = _reg_pinv(a, rank=2)
    assert loading_factor == 0
    assert rank == 2
    assert_allclose(a_inv_np, a_inv_mne, atol=1e-14)

    # Test inversion with automatic rank detection
    a_inv_mne, _, estimated_rank = _reg_pinv(a, rank=None)
    assert_allclose(a_inv_np, a_inv_mne, atol=1e-14)
    assert estimated_rank == 2

    # Test adding regularization
    a_inv_mne, loading_factor, estimated_rank = _reg_pinv(a, reg=2)
    # Since A has a diagonal of all ones, loading_factor should equal the
    # regularization parameter
    assert loading_factor == 2
    # The estimated rank should be that of the non-regularized matrix
    assert estimated_rank == 2
    # Test result against the NumPy version
    a_inv_np = np.linalg.pinv(a + loading_factor * np.eye(3), hermitian=True)
    assert_allclose(a_inv_np, a_inv_mne, atol=1e-14)

    # Test setting rcond
    a_inv_np = np.linalg.pinv(a, rcond=0.5)
    a_inv_mne, _, estimated_rank = _reg_pinv(a, rcond=0.5)
    assert_allclose(a_inv_np, a_inv_mne, atol=1e-14)
    assert estimated_rank == 1

    # Test inverting an all zero cov
    a_inv, loading_factor, estimated_rank = _reg_pinv(np.zeros((3, 3)), reg=2)
    assert_array_equal(a_inv, 0)
    assert loading_factor == 0
    assert estimated_rank == 0


def test_object_size():
    """Test object size estimation."""
    assert (object_size(np.ones(10, np.float32)) <
            object_size(np.ones(10, np.float64)))
    for lower, upper, obj in ((0, 60, ''),
                              (0, 30, 1),
                              (0, 30, 1.),
                              (0, 70, 'foo'),
                              (0, 150, np.ones(0)),
                              (0, 150, np.int32(1)),
                              (150, 500, np.ones(20)),
                              (100, 400, dict()),
                              (400, 1000, dict(a=np.ones(50))),
                              (200, 900, sparse.eye(20, format='csc')),
                              (200, 900, sparse.eye(20, format='csr'))):
        size = object_size(obj)
        assert lower < size < upper, \
            '%s < %s < %s:\n%s' % (lower, size, upper, obj)
    # views work properly
    x = dict(a=1)
    assert object_size(x) < 1000
    x['a'] = np.ones(100000, float)
    nb = x['a'].nbytes
    sz = object_size(x)
    assert nb < sz < nb * 1.01
    x['b'] = x['a']
    sz = object_size(x)
    assert nb < sz < nb * 1.01
    x['b'] = x['a'].view()
    x['b'].flags.writeable = False
    assert x['a'].flags.writeable
    sz = object_size(x)
    assert nb < sz < nb * 1.01


def test_object_diff_with_nan():
    """Test object diff can handle NaNs."""
    d0 = np.array([1, np.nan, 0])
    d1 = np.array([1, np.nan, 0])
    d2 = np.array([np.nan, 1, 0])

    assert object_diff(d0, d1) == ''
    assert object_diff(d0, d2) != ''
    assert object_diff(np.nan, np.nan) == ''
    assert object_diff(np.nan, 3.5) == ' value mismatch (nan, 3.5)\n'


def test_hash():
    """Test dictionary hashing and comparison functions."""
    # does hashing all of these types work:
    # {dict, list, tuple, ndarray, str, float, int, None}
    d0 = dict(a=dict(a=0.1, b='fo', c=1), b=[1, 'b'], c=(), d=np.ones(3),
              e=None)
    d0[1] = None
    d0[2.] = b'123'

    d1 = deepcopy(d0)
    assert len(object_diff(d0, d1)) == 0
    assert len(object_diff(d1, d0)) == 0
    assert object_hash(d0) == object_hash(d1)

    # change values slightly
    d1['data'] = np.ones(3, int)
    d1['d'][0] = 0
    assert object_hash(d0) != object_hash(d1)

    d1 = deepcopy(d0)
    assert object_hash(d0) == object_hash(d1)
    d1['a']['a'] = 0.11
    assert (len(object_diff(d0, d1)) > 0)
    assert (len(object_diff(d1, d0)) > 0)
    assert object_hash(d0) != object_hash(d1)

    d1 = deepcopy(d0)
    assert object_hash(d0) == object_hash(d1)
    d1['a']['d'] = 0  # non-existent key
    assert (len(object_diff(d0, d1)) > 0)
    assert (len(object_diff(d1, d0)) > 0)
    assert object_hash(d0) != object_hash(d1)

    d1 = deepcopy(d0)
    assert object_hash(d0) == object_hash(d1)
    d1['b'].append(0)  # different-length lists
    assert (len(object_diff(d0, d1)) > 0)
    assert (len(object_diff(d1, d0)) > 0)
    assert object_hash(d0) != object_hash(d1)

    d1 = deepcopy(d0)
    assert object_hash(d0) == object_hash(d1)
    d1['e'] = 'foo'  # non-None
    assert (len(object_diff(d0, d1)) > 0)
    assert (len(object_diff(d1, d0)) > 0)
    assert object_hash(d0) != object_hash(d1)

    d1 = deepcopy(d0)
    d2 = deepcopy(d0)
    d1['e'] = StringIO()
    d2['e'] = StringIO()
    d2['e'].write('foo')
    assert (len(object_diff(d0, d1)) > 0)
    assert (len(object_diff(d1, d0)) > 0)

    d1 = deepcopy(d0)
    d1[1] = 2
    assert (len(object_diff(d0, d1)) > 0)
    assert (len(object_diff(d1, d0)) > 0)
    assert object_hash(d0) != object_hash(d1)

    # generators (and other types) not supported
    d1 = deepcopy(d0)
    d2 = deepcopy(d0)
    d1[1] = (x for x in d0)
    d2[1] = (x for x in d0)
    pytest.raises(RuntimeError, object_diff, d1, d2)
    pytest.raises(RuntimeError, object_hash, d1)

    x = sparse.eye(2, 2, format='csc')
    y = sparse.eye(2, 2, format='csr')
    assert ('type mismatch' in object_diff(x, y))
    y = sparse.eye(2, 2, format='csc')
    assert len(object_diff(x, y)) == 0
    y[1, 1] = 2
    assert ('elements' in object_diff(x, y))
    y = sparse.eye(3, 3, format='csc')
    assert ('shape' in object_diff(x, y))
    y = 0
    assert ('type mismatch' in object_diff(x, y))

    # smoke test for gh-4796
    assert object_hash(np.int64(1)) != 0
    assert object_hash(np.bool_(True)) != 0


@requires_sklearn
@pytest.mark.parametrize('n_components', (None, 0.9999, 8, 'mle'))
@pytest.mark.parametrize('whiten', (True, False))
def test_pca(n_components, whiten):
    """Test PCA equivalence."""
    from sklearn.decomposition import PCA
    n_samples, n_dim = 1000, 10
    X = np.random.RandomState(0).randn(n_samples, n_dim)
    X[:, -1] = np.mean(X[:, :-1], axis=-1)  # true X dim is ndim - 1
    X_orig = X.copy()
    pca_skl = PCA(n_components, whiten=whiten, svd_solver='full')
    pca_mne = _PCA(n_components, whiten=whiten)
    X_skl = pca_skl.fit_transform(X)
    assert_array_equal(X, X_orig)
    X_mne = pca_mne.fit_transform(X)
    assert_array_equal(X, X_orig)
    assert_allclose(X_skl, X_mne)
    assert pca_mne.n_components_ == pca_skl.n_components_
    for key in ('mean_', 'components_',
                'explained_variance_', 'explained_variance_ratio_'):
        val_skl, val_mne = getattr(pca_skl, key), getattr(pca_mne, key)
        assert_allclose(val_skl, val_mne)
    if isinstance(n_components, float):
        assert pca_mne.n_components_ == n_dim - 1
    elif isinstance(n_components, int):
        assert pca_mne.n_components_ == n_components
    elif n_components == 'mle':
        assert pca_mne.n_components_ == n_dim - 1
    else:
        assert n_components is None
        assert pca_mne.n_components_ == n_dim


def test_array_equal_nan():
    """Test comparing arrays with NaNs."""
    a = b = [1, np.nan, 0]
    assert not np.array_equal(a, b)  # this is the annoying behavior we avoid
    assert _array_equal_nan(a, b)
    b = [np.nan, 1, 0]
    assert not _array_equal_nan(a, b)
    a = b = [np.nan] * 2
    assert _array_equal_nan(a, b)


def test_julian_conversions():
    """Test julian calendar conversions."""
    # https://aa.usno.navy.mil/data/docs/JulianDate.php
    # A.D. 1922 Jun 13  12:00:00.0  2423219.000000
    # A.D. 2018 Oct 3   12:00:00.0  2458395.000000

    jds = [2423219, 2458395, 2445701]
    dds = [datetime(1922, 6, 13, 12, 0, 0, tzinfo=timezone.utc),
           datetime(2018, 10, 3, 12, 0, 0, tzinfo=timezone.utc),
           datetime(1984, 1, 1, 12, 0, 0, tzinfo=timezone.utc)]
    cals = [(1922, 6, 13), (2018, 10, 3), (1984, 1, 1)]

    for dd, cal, jd in zip(dds, cals, jds):
        assert (dd == _julian_to_dt(jd))
        assert (cal == _julian_to_cal(jd))

        assert (jd == _dt_to_julian(dd))
        assert (jd == _cal_to_julian(cal[0], cal[1], cal[2]))


def test_grand_average_empty_sequence():
    """Test if mne.grand_average handles an empty sequence correctly."""
    with pytest.raises(ValueError, match='Please pass a list of Evoked'):
        grand_average([])


def test_grand_average_len_1():
    """Test if mne.grand_average handles a sequence of length 1 correctly."""
    # returns a list of length 1
    evokeds = read_evokeds(ave_fname, condition=[0], proj=True)

    with pytest.warns(RuntimeWarning, match='Only a single dataset'):
        gave = grand_average(evokeds)

    assert_allclose(gave.data, evokeds[0].data)


def test_reuse_cycle():
    """Test _ReuseCycle."""
    vals = 'abcde'
    iterable = _ReuseCycle(vals)
    assert ''.join(next(iterable) for _ in range(2 * len(vals))) == vals + vals
    # we're back to initial
    assert ''.join(next(iterable) for _ in range(2)) == 'ab'
    iterable.restore('a')
    assert ''.join(next(iterable) for _ in range(10)) == 'acdeabcdea'
    assert ''.join(next(iterable) for _ in range(4)) == 'bcde'
    # we're back to initial
    assert ''.join(next(iterable) for _ in range(3)) == 'abc'
    iterable.restore('a')
    iterable.restore('b')
    iterable.restore('c')
    assert ''.join(next(iterable) for _ in range(5)) == 'abcde'
    # we're back to initial
    assert ''.join(next(iterable) for _ in range(3)) == 'abc'
    iterable.restore('a')
    iterable.restore('c')
    assert ''.join(next(iterable) for _ in range(4)) == 'acde'
    assert ''.join(next(iterable) for _ in range(5)) == 'abcde'
    # we're back to initial
    assert ''.join(next(iterable) for _ in range(3)) == 'abc'
    iterable.restore('c')
    iterable.restore('a')
    with pytest.warns(RuntimeWarning, match='Could not find'):
        iterable.restore('a')
    assert ''.join(next(iterable) for _ in range(4)) == 'acde'
    assert ''.join(next(iterable) for _ in range(5)) == 'abcde'
