# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)


import os
import os.path as op
from copy import deepcopy
import warnings

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_true, assert_raises, assert_equal

from mne.fiff.bti import Raw, _read_raw, read_config, read_head_shape, Raw,\
                         read_raw_bti
from mne.fiff.bti.constants import BTI
from mne.fiff.bti.transforms import apply_trans, inverse_trans,\
                                    merge_trans

from mne.utils import _TempDir

base_dir = op.join(op.dirname(__file__), 'data')
pdf_fname = op.join(base_dir, 'test_pdf')
config_fname = op.join(base_dir, 'test_config')
config_solaris_fname = op.join(base_dir, 'test_config_solaris')
hs_fname = op.join(base_dir, 'test_hs_file')
tempdir = _TempDir()


def test_read_config():
    """ Test read bti config file """
    for config in config_fname, config_solaris_fname:
        cfg = read_config(config)
        chan_sum = sum(c['hdr']['chan_no'] for c in cfg.channels)
        check_sum = np.arange(len(cfg.channels) + 1).sum()
        assert_true(chan_sum == check_sum)


def test_read_pdf():
    """ Test read bti PDF file """
    for config in config_fname, config_solaris_fname:
        info, data = _read_raw(pdf_fname, config)
        info['fid'].close()


def test_raw():
    """ Test conversion to Raw object """
    raw = Raw(pdf_fname)
    print raw


def test_read_headshape():
    """ Test reading bti headshape """
    idx_pnts, dig_pnts = read_head_shape(hs_fname)


def test_convert_headshape():
    """ Test reading bti headshape """
    pass


def test_read_raw_bti():
    """ Test highlevel wrapper function """
    pass
