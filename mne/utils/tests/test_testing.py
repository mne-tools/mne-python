import os.path as op
import os

import numpy as np
import pytest
from numpy.testing import assert_equal

from mne.datasets import testing
from mne.io import show_fiff
from mne.utils import (_TempDir, _url_to_local_path, run_tests_if_main,
                       buggy_mkl_svd)


base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
fname_evoked = op.join(base_dir, 'test-ave.fif')
fname_raw = op.join(base_dir, 'test_raw.fif')

data_path = testing.data_path(download=False)
fname_fsaverage_trans = op.join(data_path, 'subjects', 'fsaverage', 'bem',
                                'fsaverage-trans.fif')


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


@testing.requires_testing_data
def test_datasets():
    """Test dataset config."""
    # gh-4192
    data_path = testing.data_path(download=False)
    os.environ['MNE_DATASETS_TESTING_PATH'] = op.dirname(data_path)
    assert testing.data_path(download=False) == data_path


@testing.requires_testing_data
def test_show_fiff():
    """Test show_fiff."""
    # this is not exhaustive, but hopefully bugs will be found in use
    info = show_fiff(fname_evoked)
    keys = ['FIFF_EPOCH', 'FIFFB_HPI_COIL', 'FIFFB_PROJ_ITEM',
            'FIFFB_PROCESSED_DATA', 'FIFFB_EVOKED', 'FIFF_NAVE',
            'FIFF_EPOCH']
    assert (all(key in info for key in keys))
    info = show_fiff(fname_raw, read_limit=1024)
    assert ('COORD_TRANS' in show_fiff(fname_fsaverage_trans))


def test_url_to_local_path():
    """Test URL to local path."""
    assert_equal(_url_to_local_path('http://google.com/home/why.html', '.'),
                 op.join('.', 'home', 'why.html'))


run_tests_if_main()
