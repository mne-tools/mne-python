# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import os.path as op
from nose.tools import assert_equal

from mne import create_info
from mne.io import RawArray
from mne.utils import logger, set_log_file, _TempDir


def test_apply_function_verbose():
    """Test apply function verbosity
    """
    def printer(x):
        logger.info('exec')

    n_ch = 100
    ch_names = [str(ii) for ii in range(n_ch)]
    tempdir = _TempDir()
    test_name = op.join(tempdir, 'test.log')
    set_log_file(test_name)
    try:
        raw = RawArray(np.zeros((n_ch, 10)),
                       create_info(ch_names, 1., 'eeg'))
        raw.apply_function(printer, None, np.float64, 1, verbose=False)
        with open(test_name) as fid:
            assert_equal(len(fid.readlines()), 0)
        raw.apply_function(printer, None, np.float64, 1, verbose=True)
        with open(test_name) as fid:
            assert_equal(len(fid.readlines()), n_ch)
    finally:
        set_log_file(None)
