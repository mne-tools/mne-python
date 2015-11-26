# Authors: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Alex Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD

import numpy as np

from nose.tools import assert_equal, assert_raises, assert_true
from numpy.testing import assert_array_equal
from distutils.version import LooseVersion
from scipy import signal, sparse

from mne.utils import run_tests_if_main
from mne.fixes import (_in1d, _tril_indices, _copysign, _unravel_index,
                       _Counter, _unique, _bincount, _digitize,
                       _sparse_block_diag, _matrix_rank, _meshgrid,
                       _isclose)
from mne.fixes import _firwin2 as mne_firwin2
from mne.fixes import _filtfilt as mne_filtfilt

rng = np.random.RandomState(0)


def test_counter():
    """Test Counter replacement"""
    import collections
    try:
        Counter = collections.Counter
    except Exception:
        pass
    else:
        a = Counter([1, 2, 1, 3])
        b = _Counter([1, 2, 1, 3])
        c = _Counter()
        c.update(b)
        for key, count in zip([1, 2, 3], [2, 1, 1]):
            assert_equal(a[key], b[key])
            assert_equal(a[key], c[key])


def test_unique():
    """Test unique() replacement
    """
    # skip test for np version < 1.5
    if LooseVersion(np.__version__) < LooseVersion('1.5'):
        return
    for arr in [np.array([]), rng.rand(10), np.ones(10)]:
        # basic
        assert_array_equal(np.unique(arr), _unique(arr))
        # with return_index=True
        x1, x2 = np.unique(arr, return_index=True, return_inverse=False)
        y1, y2 = _unique(arr, return_index=True, return_inverse=False)
        assert_array_equal(x1, y1)
        assert_array_equal(x2, y2)
        # with return_inverse=True
        x1, x2 = np.unique(arr, return_index=False, return_inverse=True)
        y1, y2 = _unique(arr, return_index=False, return_inverse=True)
        assert_array_equal(x1, y1)
        assert_array_equal(x2, y2)
        # with both:
        x1, x2, x3 = np.unique(arr, return_index=True, return_inverse=True)
        y1, y2, y3 = _unique(arr, return_index=True, return_inverse=True)
        assert_array_equal(x1, y1)
        assert_array_equal(x2, y2)
        assert_array_equal(x3, y3)


def test_bincount():
    """Test bincount() replacement
    """
    # skip test for np version < 1.6
    if LooseVersion(np.__version__) < LooseVersion('1.6'):
        return
    for minlength in [None, 100]:
        x = _bincount(np.ones(10, int), None, minlength)
        y = np.bincount(np.ones(10, int), None, minlength)
        assert_array_equal(x, y)


def test_in1d():
    """Test numpy.in1d() replacement"""
    a = np.arange(10)
    b = a[a % 2 == 0]
    assert_equal(_in1d(a, b).sum(), 5)


def test_digitize():
    """Test numpy.digitize() replacement"""
    data = np.arange(9)
    bins = [0, 5, 10]
    left = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2])
    right = np.array([0, 1, 1, 1, 1, 1, 2, 2, 2])

    assert_array_equal(_digitize(data, bins), left)
    assert_array_equal(_digitize(data, bins, True), right)
    assert_raises(NotImplementedError, _digitize, data + 0.1, bins, True)
    assert_raises(NotImplementedError, _digitize, data, [0., 5, 10], True)


def test_tril_indices():
    """Test numpy.tril_indices() replacement"""
    il1 = _tril_indices(4)
    il2 = _tril_indices(4, -1)

    a = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])

    assert_array_equal(a[il1],
                       np.array([1, 5, 6, 9, 10, 11, 13, 14, 15, 16]))

    assert_array_equal(a[il2], np.array([5, 9, 10, 13, 14, 15]))


def test_unravel_index():
    """Test numpy.unravel_index() replacement"""
    assert_equal(_unravel_index(2, (2, 3)), (0, 2))
    assert_equal(_unravel_index(2, (2, 2)), (1, 0))
    assert_equal(_unravel_index(254, (17, 94)), (2, 66))
    assert_equal(_unravel_index((2 * 3 + 1) * 6 + 4, (4, 3, 6)), (2, 1, 4))
    assert_array_equal(_unravel_index(np.array([22, 41, 37]), (7, 6)),
                       [[3, 6, 6], [4, 5, 1]])
    assert_array_equal(_unravel_index(1621, (6, 7, 8, 9)), (3, 1, 4, 1))


def test_copysign():
    """Test numpy.copysign() replacement"""
    a = np.array([-1, 1, -1])
    b = np.array([1, -1, 1])

    assert_array_equal(_copysign(a, b), b)
    assert_array_equal(_copysign(b, a), a)


def test_firwin2():
    """Test firwin2 backport
    """
    taps1 = mne_firwin2(150, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
    taps2 = signal.firwin2(150, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
    assert_array_equal(taps1, taps2)


def test_filtfilt():
    """Test IIR filtfilt replacement
    """
    x = np.r_[1, np.zeros(100)]
    # Filter with an impulse
    y = mne_filtfilt([1, 0], [1, 0], x, padlen=0)
    assert_array_equal(x, y)


def test_sparse_block_diag():
    """Test sparse block diag replacement"""
    x = _sparse_block_diag([sparse.eye(2, 2), sparse.eye(2, 2)])
    x = x - sparse.eye(4, 4)
    x.eliminate_zeros()
    assert_equal(len(x.data), 0)


def test_rank():
    """Test rank replacement"""
    assert_equal(_matrix_rank(np.ones(10)), 1)
    assert_equal(_matrix_rank(np.eye(10)), 10)
    assert_equal(_matrix_rank(np.ones((10, 10))), 1)
    assert_raises(TypeError, _matrix_rank, np.ones((10, 10, 10)))


def test_meshgrid():
    """Test meshgrid replacement
    """
    a = np.arange(10)
    b = np.linspace(0, 1, 5)
    a_grid, b_grid = _meshgrid(a, b, indexing='ij')
    for grid in (a_grid, b_grid):
        assert_equal(grid.shape, (a.size, b.size))
    a_grid, b_grid = _meshgrid(a, b, indexing='xy', copy=True)
    for grid in (a_grid, b_grid):
        assert_equal(grid.shape, (b.size, a.size))
    assert_raises(TypeError, _meshgrid, a, b, foo='a')
    assert_raises(ValueError, _meshgrid, a, b, indexing='foo')


def test_isclose():
    """Test isclose replacement
    """
    a = rng.randn(10)
    b = a.copy()
    assert_true(_isclose(a, b).all())
    a[0] = np.inf
    b[0] = np.inf
    a[-1] = np.nan
    b[-1] = np.nan
    assert_true(_isclose(a, b, equal_nan=True).all())

run_tests_if_main()
