# -*- coding: utf-8 -*-
from os import path as op
from nose.tools import assert_raises, assert_true, assert_equal

import numpy as np
from scipy import sparse

from mne._hdf5 import write_hdf5, read_hdf5
from mne.utils import requires_h5py, _TempDir, object_diff, run_tests_if_main


@requires_h5py
def test_hdf5():
    """Test HDF5 IO
    """
    tempdir = _TempDir()
    test_file = op.join(tempdir, 'test.hdf5')
    sp = sparse.eye(3, 3, format='csc')
    sp[2, 2] = 2
    x = dict(a=dict(b=np.zeros(3)), c=np.zeros(2, np.complex128),
             d=[dict(e=(1, -2., 'hello', u'goodbyeu\u2764')), None], f=sp)
    write_hdf5(test_file, 1)
    assert_equal(read_hdf5(test_file), 1)
    assert_raises(IOError, write_hdf5, test_file, x)  # file exists
    write_hdf5(test_file, x, overwrite=True)
    assert_raises(IOError, read_hdf5, test_file + 'FOO')  # not found
    xx = read_hdf5(test_file)
    assert_true(object_diff(x, xx) == '')  # no assert_equal, ugly output


run_tests_if_main()
