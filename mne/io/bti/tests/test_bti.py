from __future__ import print_function
# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os
import os.path as op
from functools import reduce

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose)
from nose.tools import assert_true, assert_raises, assert_equal

from mne.io import Raw as Raw
from mne.io.bti.bti import (_read_config, _setup_head_shape,
                            _read_data, _read_bti_header)
from mne.io import read_raw_bti
from mne import concatenate_raws
from mne.utils import run_tests_if_main


base_dir = op.join(op.abspath(op.dirname(__file__)), 'data')

archs = 'linux', 'solaris'
pdf_fnames = [op.join(base_dir, 'test_pdf_%s' % a) for a in archs]
config_fnames = [op.join(base_dir, 'test_config_%s' % a) for a in archs]
hs_fnames = [op.join(base_dir, 'test_hs_%s' % a) for a in archs]
exported_fnames = [op.join(base_dir, 'exported4D_%s_raw.fif' % a)
                   for a in archs]
tmp_raw_fname = op.join(base_dir, 'tmp_raw.fif')

# the 4D exporter doesn't export all channels, so we confine our comparison
NCH = 248


def test_read_config():
    """ Test read bti config file """
    # for config in config_fname, config_solaris_fname:
    for config in config_fnames:
        cfg = _read_config(config)
        assert_true(all('unknown' not in block.lower() and block != ''
                        for block in cfg['user_blocks']))


def test_read_pdf():
    """ Test read bti PDF file """
    for pdf, config in zip(pdf_fnames, config_fnames):
        info = _read_bti_header(pdf, config)
        data = _read_data(info)
        shape = (info['total_chans'], info['total_slices'])
        assert_true(data.shape == shape)


def test_crop_append():
    """ Test crop and append raw """
    raw = read_raw_bti(pdf_fnames[0], config_fnames[0], hs_fnames[0])
    raw.preload_data()  # currently does nothing
    y, t = raw[:]
    t0, t1 = 0.25 * t[-1], 0.75 * t[-1]
    mask = (t0 <= t) * (t <= t1)
    raw_ = raw.crop(t0, t1)
    y_, _ = raw_[:]
    assert_true(y_.shape[1] == mask.sum())
    assert_true(y_.shape[0] == y.shape[0])

    raw2 = raw.copy()
    assert_raises(RuntimeError, raw.append, raw2, preload=False)
    raw.append(raw2)
    assert_allclose(np.tile(raw2[:, :][0], (1, 2)), raw[:, :][0])


def test_raw():
    """ Test bti conversion to Raw object """

    for pdf, config, hs, exported in zip(pdf_fnames, config_fnames, hs_fnames,
                                         exported_fnames):
        # rx = 2 if 'linux' in pdf else 0
        assert_raises(ValueError, read_raw_bti, pdf, 'eggs')
        assert_raises(ValueError, read_raw_bti, pdf, config, 'spam')
        if op.exists(tmp_raw_fname):
            os.remove(tmp_raw_fname)
        with Raw(exported, preload=True) as ex:
            with read_raw_bti(pdf, config, hs) as ra:
                assert_true('RawBTi' in repr(ra))
                assert_equal(ex.ch_names[:NCH], ra.ch_names[:NCH])
                assert_array_almost_equal(ex.info['dev_head_t']['trans'],
                                          ra.info['dev_head_t']['trans'], 7)
                dig1, dig2 = [np.array([d['r'] for d in r_.info['dig']])
                              for r_ in (ra, ex)]
                assert_array_equal(dig1, dig2)

                coil1, coil2 = [np.concatenate([d['coil_trans'].flatten()
                                for d in r_.info['chs'][:NCH]])
                                for r_ in (ra, ex)]
                assert_array_almost_equal(coil1, coil2, 7)

                loc1, loc2 = [np.concatenate([d['loc'].flatten()
                              for d in r_.info['chs'][:NCH]])
                              for r_ in (ra, ex)]
                assert_array_equal(loc1, loc2)

                assert_array_equal(ra._data[:NCH], ex._data[:NCH])
                assert_array_equal(ra._cals[:NCH], ex._cals[:NCH])

                # check our transforms
                for key in ('dev_head_t', 'dev_ctf_t', 'ctf_head_t'):
                    if ex.info[key] is None:
                        pass  # MNE-C doesn't convert all fields
                    else:
                        assert_true(ra.info[key] is not None)
                        for ent in ('to', 'from', 'trans'):
                            assert_allclose(ex.info[key][ent],
                                            ra.info[key][ent])

                # Make sure concatenation works
                raw_concat = concatenate_raws([ra.copy(), ra])
                assert_equal(raw_concat.n_times, 2 * ra.n_times)

                ra.save(tmp_raw_fname)
            with Raw(tmp_raw_fname) as r:
                print(r)
                for key in ('dev_head_t', 'dev_ctf_t', 'ctf_head_t'):
                    assert_true(isinstance(r.info[key], dict))
                    assert_equal(r.info[key]['trans'].shape, (4, 4))
        os.remove(tmp_raw_fname)


def test_setup_headshape():
    """ Test reading bti headshape """
    for hs in hs_fnames:
        dig, t = _setup_head_shape(hs)
        expected = set(['kind', 'ident', 'r'])
        found = set(reduce(lambda x, y: list(x) + list(y),
                           [d.keys() for d in dig]))
        assert_true(not expected - found)

run_tests_if_main()
