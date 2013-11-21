# Authors: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Alex Gramfort <gramfort@nmr.mgh.harvard.edu>
# License: BSD

import numpy as np

from nose.tools import assert_equal
from numpy.testing import assert_array_equal
from scipy import signal

from ..fixes import _in1d, _tril_indices, _copysign, _unravel_index
from ..fixes import _firwin2 as mne_firwin2
from ..fixes import _filtfilt as mne_filtfilt


def test_in1d():
    """Test numpy.in1d() replacement"""
    a = np.arange(10)
    b = a[a % 2 == 0]
    assert_equal(_in1d(a, b).sum(), 5)


def test_tril_indices():
    """Test numpy.tril_indices() replacement"""
    il1 = _tril_indices(4)
    il2 = _tril_indices(4, -1)

    a = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])

    assert_array_equal(a[il1],
                       np.array([1,  5,  6,  9, 10, 11, 13, 14, 15, 16]))

    assert_array_equal(a[il2], np.array([5, 9, 10, 13, 14, 15]))


def test_unravel_index():
    """Test numpy.unravel_index() replacement"""
    assert_equal(_unravel_index(2, (2, 3)), (0, 2))
    assert_equal(_unravel_index(2,(2,2)), (1,0))
    assert_equal(_unravel_index(254,(17,94)), (2,66))
    assert_equal(_unravel_index((2*3 + 1)*6 + 4, (4,3,6)), (2,1,4))
    assert_array_equal(_unravel_index(np.array([22, 41, 37]), (7,6)),
                    [[3, 6, 6],[4, 5, 1]])
    assert_array_equal(_unravel_index(1621, (6,7,8,9)), (3,1,4,1))


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
