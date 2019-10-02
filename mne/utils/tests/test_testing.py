import os.path as op
import os

import numpy as np
import pytest
from numpy.testing import assert_equal

from mne.datasets import testing
from mne.utils import (_TempDir, _url_to_local_path, run_tests_if_main,
                       buggy_mkl_svd)


def test_buggy_mkl():
    """Test decorator for buggy MKL issues."""
    from unittest import SkipTest

    @buggy_mkl_svd
    def foo(a, b):
        raise np.linalg.LinAlgError('SVD did not converge')
    with pytest.warns(RuntimeWarning, match='convergence error'):
        pytest.raises(SkipTest, foo, 1, 2)

    @buggy_mkl_svd
    def bar(c, d, e):
        raise RuntimeError('SVD did not converge')
    pytest.raises(RuntimeError, bar, 1, 2, 3)


def test_tempdir():
    """Test TempDir."""
    tempdir2 = _TempDir()
    assert (op.isdir(tempdir2))
    x = str(tempdir2)
    del tempdir2
    assert (not op.isdir(x))


def test_datasets():
    """Test dataset config."""
    # gh-4192
    data_path = testing.data_path(download=False)
    os.environ['MNE_DATASETS_TESTING_PATH'] = op.dirname(data_path)
    assert testing.data_path(download=False) == data_path


def test_url_to_local_path():
    """Test URL to local path."""
    assert_equal(_url_to_local_path('http://google.com/home/why.html', '.'),
                 op.join('.', 'home', 'why.html'))


run_tests_if_main()
