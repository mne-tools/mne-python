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
from mne.io.bti.bti import (_read_config, _process_bti_headshape,
                            _read_data, _read_bti_header, _get_bti_dev_t,
                            _correct_trans, _get_bti_info)
from mne.io import read_raw_bti
from mne.io.constants import FIFF
from mne import concatenate_raws
from mne.utils import run_tests_if_main
from mne.transforms import Transform, combine_transforms, invert_transform
from mne.externals import six

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
    raw.load_data()  # currently does nothing
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


def test_transforms():
    """ Test transformations """
    bti_trans = (0.0, 0.02, 0.11)
    bti_dev_t = Transform('ctf_meg', 'meg', _get_bti_dev_t(0.0, bti_trans))
    for pdf, config, hs, in zip(pdf_fnames, config_fnames, hs_fnames):
        raw = read_raw_bti(pdf, config, hs)
        dev_ctf_t = raw.info['dev_ctf_t']
        dev_head_t_old = raw.info['dev_head_t']
        ctf_head_t = raw.info['ctf_head_t']

        # 1) get BTI->Neuromag
        bti_dev_t = Transform('ctf_meg', 'meg', _get_bti_dev_t(0.0, bti_trans))

        # 2) get Neuromag->BTI head
        t = combine_transforms(invert_transform(bti_dev_t), dev_ctf_t,
                               'meg', 'ctf_head')
        # 3) get Neuromag->head
        dev_head_t_new = combine_transforms(t, ctf_head_t, 'meg', 'head')

        assert_array_equal(dev_head_t_new['trans'], dev_head_t_old['trans'])


def test_raw():
    """ Test bti conversion to Raw object """
    for pdf, config, hs, exported in zip(pdf_fnames, config_fnames, hs_fnames,
                                         exported_fnames):
        # rx = 2 if 'linux' in pdf else 0
        assert_raises(ValueError, read_raw_bti, pdf, 'eggs')
        assert_raises(ValueError, read_raw_bti, pdf, config, 'spam')
        if op.exists(tmp_raw_fname):
            os.remove(tmp_raw_fname)
        ex = Raw(exported, preload=True)
        ra = read_raw_bti(pdf, config, hs)
        assert_true('RawBTi' in repr(ra))
        assert_equal(ex.ch_names[:NCH], ra.ch_names[:NCH])
        assert_array_almost_equal(ex.info['dev_head_t']['trans'],
                                  ra.info['dev_head_t']['trans'], 7)
        dig1, dig2 = [np.array([d['r'] for d in r_.info['dig']])
                      for r_ in (ra, ex)]
        assert_array_almost_equal(dig1, dig2, 18)
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
                pass
            else:
                assert_true(ra.info[key] is not None)
                for ent in ('to', 'from', 'trans'):
                    assert_allclose(ex.info[key][ent],
                                    ra.info[key][ent])

        # Make sure concatenation works
        raw_concat = concatenate_raws([ra.copy(), ra])
        assert_equal(raw_concat.n_times, 2 * ra.n_times)

        ra.save(tmp_raw_fname)
        re = Raw(tmp_raw_fname)
        print(re)
        for key in ('dev_head_t', 'dev_ctf_t', 'ctf_head_t'):
            assert_true(isinstance(re.info[key], dict))
            this_t = re.info[key]['trans']
            assert_equal(this_t.shape, (4, 4))
            # cehck that matrix by is not identity
            assert_true(not np.allclose(this_t, np.eye(4)))
        os.remove(tmp_raw_fname)


def test_info_no_rename_no_reorder():
    """ Test private renaming and reordering option """
    for pdf, config, hs in zip(pdf_fnames, config_fnames, hs_fnames):
        info, bti_info = _get_bti_info(
            pdf_fname=pdf, config_fname=config, head_shape_fname=hs,
            rotation_x=0.0, translation=(0.0, 0.02, 0.11), convert=False,
            ecg_ch='E31', eog_ch=('E63', 'E64'),
            rename_channels=False, sort_by_ch_name=False)
        assert_equal(info['ch_names'],
                     [ch['ch_name'] for ch in info['chs']])
        assert_equal([n for n in info['ch_names'] if n.startswith('A')][:5],
                     ['A22', 'A2', 'A104', 'A241', 'A138'])
        assert_equal([n for n in info['ch_names'] if n.startswith('A')][-5:],
                     ['A133', 'A158', 'A44', 'A134', 'A216'])


def test_no_conversion():
    """ Test bti no-conversion option """
    for pdf, config, hs in zip(pdf_fnames, config_fnames, hs_fnames):
        raw = read_raw_bti(pdf, config, hs, convert=False)
        raw_con = read_raw_bti(pdf, config, hs, convert=True)

        bti_info = _read_bti_header(pdf, config)
        dev_ctf_t = _correct_trans(bti_info['bti_transform'][0])
        assert_array_equal(dev_ctf_t, raw.info['dev_ctf_t']['trans'])
        assert_array_equal(raw.info['dev_head_t']['trans'], np.eye(4))
        assert_array_equal(raw.info['ctf_head_t']['trans'], np.eye(4))
        dig, t = _process_bti_headshape(hs, convert=False, use_hpi=False)
        assert_array_equal(t['trans'], np.eye(4))

        for ii, (old, new, con) in enumerate(zip(
                dig, raw.info['dig'], raw_con.info['dig'])):
            assert_equal(old['ident'], new['ident'])
            assert_array_equal(old['r'], new['r'])
            assert_true(not np.allclose(old['r'], con['r']))

            if ii > 10:
                break

        ch_map = dict((ch['chan_label'],
                       ch['coil_trans']) for ch in bti_info['chs'])

        for ii, ch_label in enumerate(raw.bti_ch_labels):
            if not ch_label.startswith('A'):
                continue
            t1 = _correct_trans(ch_map[ch_label])
            t2 = raw.info['chs'][ii]['coil_trans']
            t3 = raw_con.info['chs'][ii]['coil_trans']
            assert_array_equal(t1, t2)
            assert_true(not np.allclose(t1, t3))
            assert_equal(
                raw_con.info['chs'][0]['coord_frame'],
                FIFF.FIFFV_COORD_DEVICE)
            assert_equal(
                raw.info['chs'][0]['coord_frame'],
                FIFF.FIFFV_MNE_COORD_4D_HEAD)


def test_bytes_io():
    """ Test bti bytes-io API """
    for pdf, config, hs in zip(pdf_fnames, config_fnames, hs_fnames):
        raw = read_raw_bti(pdf, config, hs, convert=True)

        with open(pdf, 'rb') as fid:
            pdf = six.BytesIO(fid.read())
        with open(config, 'rb') as fid:
            config = six.BytesIO(fid.read())
        with open(hs, 'rb') as fid:
            hs = six.BytesIO(fid.read())
        raw2 = read_raw_bti(pdf, config, hs, convert=True)
        repr(raw2)
        assert_array_equal(raw._data, raw2._data)


def test_setup_headshape():
    """ Test reading bti headshape """
    for hs in hs_fnames:
        dig, t = _process_bti_headshape(hs)
        expected = set(['kind', 'ident', 'r'])
        found = set(reduce(lambda x, y: list(x) + list(y),
                           [d.keys() for d in dig]))
        assert_true(not expected - found)

run_tests_if_main()
