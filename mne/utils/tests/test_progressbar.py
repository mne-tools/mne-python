import os.path as op

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from mne.parallel import parallel_func
from mne.utils import ProgressBar, array_split_idx


def test_progressbar():
    """Test progressbar class."""
    a = np.arange(10)
    pbar = ProgressBar(a)
    assert a is pbar.iterable
    assert pbar.max_value == 10

    pbar = ProgressBar(10)
    assert pbar.max_value == 10
    assert pbar.iterable is None

    # Make sure that non-iterable input raises an error
    def iter_func(a):
        for ii in a:
            pass
    pytest.raises(ValueError, iter_func, ProgressBar(20))


def _identity(x):
    return x


def test_progressbar_parallel_basic(capsys):
    """Test ProgressBar with parallel computing, basic version."""
    assert capsys.readouterr().out == ''
    parallel, p_fun, _ = parallel_func(_identity, total=10, n_jobs=1,
                                       verbose=True)
    out = parallel(p_fun(x) for x in range(10))
    assert out == list(range(10))
    assert '100.00%' in capsys.readouterr().out


def _identity_block(x, pb):
    for ii in range(len(x)):
        pb.update(ii + 1)
    return x


def test_progressbar_parallel_advanced(capsys):
    """Test ProgressBar with parallel computing, advanced version."""
    assert capsys.readouterr().out == ''
    # This must be "1" because "capsys" won't get stdout properly otherwise
    parallel, p_fun, _ = parallel_func(_identity_block, n_jobs=1,
                                       verbose=False)
    arr = np.arange(10)
    with ProgressBar(len(arr), verbose_bool=True) as pb:
        out = parallel(p_fun(x, pb.subset(pb_idx))
                       for pb_idx, x in array_split_idx(arr, 2))
        assert op.isfile(pb._mmap_fname)
        sum_ = np.memmap(pb._mmap_fname, dtype='bool', mode='r',
                         shape=10).sum()
        assert sum_ == len(arr)
    assert not op.isfile(pb._mmap_fname), '__exit__ not called?'
    out = np.concatenate(out)
    assert_array_equal(out, arr)
    assert '100.00%' in capsys.readouterr().out


def _identity_block_wide(x, pb):
    for ii in range(len(x)):
        for jj in range(2):
            pb.update(ii * 2 + jj + 1)
    return x, pb.idx


def test_progressbar_parallel_more(capsys):
    """Test ProgressBar with parallel computing, advanced version."""
    assert capsys.readouterr().out == ''
    # This must be "1" because "capsys" won't get stdout properly otherwise
    parallel, p_fun, _ = parallel_func(_identity_block_wide, n_jobs=1,
                                       verbose=False)
    arr = np.arange(10)
    with ProgressBar(len(arr) * 2, verbose_bool=True) as pb:
        out = parallel(p_fun(x, pb.subset(pb_idx))
                       for pb_idx, x in array_split_idx(arr, 2, n_per_split=2))
        idxs = np.concatenate([o[1] for o in out])
        assert_array_equal(idxs, np.arange(len(arr) * 2))
        out = np.concatenate([o[0] for o in out])
        assert op.isfile(pb._mmap_fname)
        sum_ = np.memmap(pb._mmap_fname, dtype='bool', mode='r',
                         shape=len(arr) * 2).sum()
        assert sum_ == len(arr) * 2
    assert not op.isfile(pb._mmap_fname), '__exit__ not called?'
    assert '100.00%' in capsys.readouterr().out
