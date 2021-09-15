import os.path as op

import numpy as np
import pytest

from mne.datasets import testing
from mne.utils import (_TempDir, _url_to_local_path, buggy_mkl_svd)


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


def test_datasets(monkeypatch, tmpdir):
    """Test dataset config."""
    # gh-4192
    fake_path = tmpdir.mkdir('MNE-testing-data')
    with open(fake_path.join('version.txt'), 'w') as fid:
        fid.write('9999.9999')
    monkeypatch.setenv('_MNE_FAKE_HOME_DIR', str(tmpdir))
    monkeypatch.setenv('MNE_DATASETS_TESTING_PATH', str(tmpdir))
    assert testing.data_path(download=False, verbose='debug') == str(fake_path)


def test_url_to_local_path():
    """Test URL to local path."""
    assert _url_to_local_path('http://google.com/home/why.html', '.') == \
        op.join('.', 'home', 'why.html')
