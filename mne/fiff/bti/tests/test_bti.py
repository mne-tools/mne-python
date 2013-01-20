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

from mne.fiff import Raw as Raw_

from mne.fiff.bti.raw import Raw, read_config, setup_head_shape,\
                             convert_coord_frame, read_raw_bti, _read_data
from mne.fiff.bti.constants import BTI
from mne.fiff.bti.transforms import apply_trans, inverse_trans,\
                                    merge_trans

from mne.utils import _TempDir

base_dir = op.join(op.abspath(op.dirname(__file__)), 'data')
pdf_fname = op.join(base_dir, 'test_pdf_file')
config_linux_fname = op.join(base_dir, 'test_config_linux')
config_solaris_fname = op.join(base_dir, 'test_config_solaris')
config_fnames = [config_solaris_fname]
hs_fname = op.join(base_dir, 'test_hs_file')
tmp_raw_fname = op.join(base_dir, 'tmp_raw.fif')
tempdir = _TempDir()


def test_read_config():
    """ Test read bti config file """
    # for config in config_fname, config_solaris_fname:
    read_config(config_solaris_fname)


def test_read_pdf():
    """ Test read bti PDF file """
    for config in config_fnames:
        info, data = _read_data(pdf_fname, config)
        info['fid'].close()


def test_raw():
    """ Test conversion to Raw object """
    assert_raises(ValueError, Raw, pdf_fname, 'eggs')
    assert_raises(ValueError, Raw, pdf_fname,
                  config_solaris_fname, 'spam')
    if op.exists(tmp_raw_fname):
        os.remove(tmp_raw_fname)
    with Raw(pdf_fname, config_solaris_fname, hs_fname) as r:
        r.save(tmp_raw_fname)
        print r.info['nchan']
    with Raw_(tmp_raw_fname) as r:
        print r
    os.remove(tmp_raw_fname)


def test_setup_headshape():
    """ Test reading bti headshape """
    dig = setup_head_shape(hs_fname)
    expected = set(['kind', 'ident', 'r'])
    found = set(reduce(lambda x, y: x + y, [d.keys() for d in dig]))
    assert_true(not expected - found)


def test_read_raw_bti():
    """ Test highlevel wrapper function """
    read_raw_bti(pdf_fname, config_solaris_fname, hs_fname)
