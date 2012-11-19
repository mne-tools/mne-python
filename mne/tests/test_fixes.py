# Authors: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Alex Gramfort <gramfort@nmr.mgh.harvard.edu>
# License: BSD

import numpy as np

from nose.tools import assert_equal
from numpy.testing import assert_array_equal

from ..fixes import _in1d, _copysign, _unravel_index


def test_in1d():
    """Test numpy.in1d() replacement"""
    a = np.arange(10)
    b = a[a % 2 == 0]
    assert_equal(_in1d(a, b).sum(), 5)


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
