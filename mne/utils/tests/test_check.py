import numpy as np
from numpy.testing import assert_array_equal
import pytest

from mne.utils import (check_random_state, _check_fname, check_fname,
                       _check_subject, requires_mayavi, traits_test,
                       _check_mayavi_version)


def test_check():
    """Test checking functions."""
    pytest.raises(ValueError, check_random_state, 'foo')
    pytest.raises(TypeError, _check_fname, 1)
    pytest.raises(IOError, check_fname, 'foo', 'tets-dip.x', (), ('.fif',))
    pytest.raises(ValueError, _check_subject, None, None)
    pytest.raises(TypeError, _check_subject, None, 1)
    pytest.raises(TypeError, _check_subject, 1, None)


@requires_mayavi
@traits_test
def test_check_mayavi():
    """Test mayavi version check."""
    pytest.raises(RuntimeError, _check_mayavi_version, '100.0.0')
