# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import os
import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_true, assert_raises, assert_equal

from mne.fiff import Raw as Raw

from mne.fiff.bti.raw import _read_config, _setup_head_shape,\
                             read_raw_bti, _read_data, _read_bti_header
from mne.fiff.bti.transforms import apply_trans, inverse_trans,\
                                    merge_trans

from mne.utils import _TempDir

base_dir = op.join(op.abspath(op.dirname(__file__)), 'data')

archs = 'linux', 'solaris'
pdf_fnames = [op.join(base_dir, 'test_pdf_%s' % a) for a in archs]
config_fnames = [op.join(base_dir, 'test_config_%s' % a) for a in archs]
hs_fnames = [op.join(base_dir, 'test_hs_%s' % a) for a in archs]
exported_fnames = [op.join(base_dir, 'exported4D_%s.fif' % a) for a in archs]
tmp_raw_fname = op.join(base_dir, 'tmp_raw.fif')
tempdir = _TempDir()

# the 4D exporter doesn't export all channels, so we confine our comparison
NCH = 248


def test_read_config():
    """ Test read bti config file """
    # for config in config_fname, config_solaris_fname:
    for config in config_fnames:
        _read_config(config)


def test_read_pdf():
    """ Test read bti PDF file """
    for pdf, config in zip(pdf_fnames, config_fnames):
        info = _read_bti_header(pdf, config)
        data = _read_data(info)
        info['fid'].close()


def test_raw():
    """ Test conversion to Raw object """

    for pdf, config, hs, exported in zip(pdf_fnames, config_fnames, hs_fnames,
                               exported_fnames):
        print pdf
        assert_raises(ValueError, read_raw_bti, pdf, 'eggs')
        assert_raises(ValueError, read_raw_bti, pdf, config, 'spam')
        if op.exists(tmp_raw_fname):
            os.remove(tmp_raw_fname)
        with Raw(exported, preload=True) as e:
            with read_raw_bti(pdf, config, hs) as r:
                ntsl = len(r._times)  # reference channels yield longer files
                assert_array_almost_equal(e.info['dev_head_t'],
                                          r.info['dev_head_t'])
                dig1, dig2 = [np.array([d['r'] for d in r_.info['dig']])
                              for r_ in r, e]
                assert_array_almost_equal(dig1, dig2)

                coil1, coil2 = [np.array([d['coil_trans'] for d in
                                r_.info['chs']]) for r_ in r, e]
                assert_array_almost_equal(coil1, coil2)
                assert_array_almost_equal(r._data[:NCH], e._data[:NCH, ntsl])
                assert_true(r.ch_names[:NCH] == e.ch_names)
                assert_true(r.cals[:NCH] == e.cals)
                r.save(tmp_raw_fname)
            with Raw(tmp_raw_fname) as r:
                print r
        os.remove(tmp_raw_fname)


def test_setup_headshape():
    """ Test reading bti headshape """
    for hs in hs_fnames:
        dig, t = _setup_head_shape(hs)
        expected = set(['kind', 'ident', 'r'])
        found = set(reduce(lambda x, y: x + y, [d.keys() for d in dig]))
        assert_true(not expected - found)
