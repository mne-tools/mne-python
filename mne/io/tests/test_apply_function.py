# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import os.path as op
from nose.tools import assert_equal, assert_raises

from mne import create_info
from mne.io import RawArray
from mne.utils import logger, set_log_file, slow_test, _TempDir


def bad_1(x):
    return  # bad return type


def bad_2(x):
    return x[:-1]  # bad shape


def printer(x):
    logger.info('exec')
    return x


@slow_test
def test_apply_function_verbose():
    """Test apply function verbosity
    """
    n_chan = 2
    n_times = 3
    ch_names = [str(ii) for ii in range(n_chan)]
    raw = RawArray(np.zeros((n_chan, n_times)),
                   create_info(ch_names, 1., 'mag'))
    # test return types in both code paths (parallel / 1 job)
    assert_raises(TypeError, raw.apply_function, bad_1,
                  None, None, 1)
    assert_raises(ValueError, raw.apply_function, bad_2,
                  None, None, 1)
    assert_raises(TypeError, raw.apply_function, bad_1,
                  None, None, 2)
    assert_raises(ValueError, raw.apply_function, bad_2,
                  None, None, 2)

    # check our arguments
    tempdir = _TempDir()
    test_name = op.join(tempdir, 'test.log')
    set_log_file(test_name)
    try:
        raw.apply_function(printer, None, None, 1, verbose=False)
        with open(test_name) as fid:
            assert_equal(len(fid.readlines()), 0)
        raw.apply_function(printer, None, None, 1, verbose=True)
        with open(test_name) as fid:
            assert_equal(len(fid.readlines()), n_chan)
    finally:
        set_log_file(None)
