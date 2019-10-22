# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

from io import BytesIO
import os
import os.path as op
from functools import reduce, partial

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose, assert_equal)
import pytest

from mne.datasets import testing
from mne.io import read_raw_fif, read_raw_bti
from mne.io._digitization import _make_bti_dig_points
from mne.io.bti.bti import (_read_config,
                            _read_bti_header, _get_bti_dev_t,
                            _correct_trans, _get_bti_info,
                            _loc_to_coil_trans, _convert_coil_trans,
                            _check_nan_dev_head_t, _rename_channels)
from mne.io.bti.bti import _read_head_shape
from mne.io.tests.test_raw import _test_raw_reader
from mne.io.pick import pick_info
from mne.io.constants import FIFF
from mne import pick_types
from mne.utils import assert_dig_allclose, run_tests_if_main
from mne.transforms import Transform, combine_transforms, invert_transform

base_dir = op.join(op.abspath(op.dirname(__file__)), 'data')

archs = 'linux', 'solaris'
pdf_fnames = [op.join(base_dir, 'test_pdf_%s' % a) for a in archs]
config_fnames = [op.join(base_dir, 'test_config_%s' % a) for a in archs]
hs_fnames = [op.join(base_dir, 'test_hs_%s' % a) for a in archs]
exported_fnames = [op.join(base_dir, 'exported4D_%s_raw.fif' % a)
                   for a in archs]
tmp_raw_fname = op.join(base_dir, 'tmp_raw.fif')

fname_2500 = op.join(testing.data_path(download=False), 'BTi', 'erm_HFH',
                     'c,rfDC')
fname_sim = op.join(testing.data_path(download=False), 'BTi', '4Dsim',
                    'c,rfDC')
fname_sim_filt = op.join(testing.data_path(download=False), 'BTi', '4Dsim',
                         'c,rfDC,fn50,o')

# the 4D exporter doesn't export all channels, so we confine our comparison
NCH = 248


@testing.requires_testing_data
def test_read_2500():
    """Test reading data from 2500 system."""
    _test_raw_reader(read_raw_bti, pdf_fname=fname_2500, head_shape_fname=None)


def test_read_config():
    """Test read bti config file."""
    # for config in config_fname, config_solaris_fname:
    for config in config_fnames:
        cfg = _read_config(config)
        assert all('unknown' not in block.lower() and block != ''
                   for block in cfg['user_blocks'])


def test_crop_append():
    """Test crop and append raw."""
    raw = _test_raw_reader(
        read_raw_bti, pdf_fname=pdf_fnames[0],
        config_fname=config_fnames[0], head_shape_fname=hs_fnames[0])
    y, t = raw[:]
    t0, t1 = 0.25 * t[-1], 0.75 * t[-1]
    mask = (t0 <= t) * (t <= t1)
    raw_ = raw.copy().crop(t0, t1)
    y_, _ = raw_[:]
    assert (y_.shape[1] == mask.sum())
    assert (y_.shape[0] == y.shape[0])


def test_transforms():
    """Test transformations."""
    bti_trans = (0.0, 0.02, 0.11)
    bti_dev_t = Transform('ctf_meg', 'meg', _get_bti_dev_t(0.0, bti_trans))
    for pdf, config, hs, in zip(pdf_fnames, config_fnames, hs_fnames):
        raw = read_raw_bti(pdf, config, hs, preload=False)
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


@pytest.mark.slowtest
def test_raw():
    """Test bti conversion to Raw object."""
    for pdf, config, hs, exported in zip(pdf_fnames, config_fnames, hs_fnames,
                                         exported_fnames):
        # rx = 2 if 'linux' in pdf else 0
        pytest.raises(ValueError, read_raw_bti, pdf, 'eggs', preload=False)
        pytest.raises(ValueError, read_raw_bti, pdf, config, 'spam',
                      preload=False)
        if op.exists(tmp_raw_fname):
            os.remove(tmp_raw_fname)
        ex = read_raw_fif(exported, preload=True)
        ra = read_raw_bti(pdf, config, hs, preload=False)
        assert ('RawBTi' in repr(ra))
        assert_equal(ex.ch_names[:NCH], ra.ch_names[:NCH])
        assert_array_almost_equal(ex.info['dev_head_t']['trans'],
                                  ra.info['dev_head_t']['trans'], 7)
        assert len(ex.info['dig']) in (3563, 5154)
        assert_dig_allclose(ex.info, ra.info, limit=100)
        coil1, coil2 = [np.concatenate([d['loc'].flatten()
                                        for d in r_.info['chs'][:NCH]])
                        for r_ in (ra, ex)]
        assert_array_almost_equal(coil1, coil2, 7)

        loc1, loc2 = [np.concatenate([d['loc'].flatten()
                                      for d in r_.info['chs'][:NCH]])
                      for r_ in (ra, ex)]
        assert_allclose(loc1, loc2)

        assert_allclose(ra[:NCH][0], ex[:NCH][0])
        assert_array_equal([c['range'] for c in ra.info['chs'][:NCH]],
                           [c['range'] for c in ex.info['chs'][:NCH]])
        assert_array_equal([c['cal'] for c in ra.info['chs'][:NCH]],
                           [c['cal'] for c in ex.info['chs'][:NCH]])
        assert_array_equal(ra._cals[:NCH], ex._cals[:NCH])

        # check our transforms
        for key in ('dev_head_t', 'dev_ctf_t', 'ctf_head_t'):
            if ex.info[key] is None:
                pass
            else:
                assert (ra.info[key] is not None)
                for ent in ('to', 'from', 'trans'):
                    assert_allclose(ex.info[key][ent],
                                    ra.info[key][ent])

        ra.save(tmp_raw_fname)
        re = read_raw_fif(tmp_raw_fname)
        print(re)
        for key in ('dev_head_t', 'dev_ctf_t', 'ctf_head_t'):
            assert (isinstance(re.info[key], dict))
            this_t = re.info[key]['trans']
            assert_equal(this_t.shape, (4, 4))
            # check that matrix by is not identity
            assert (not np.allclose(this_t, np.eye(4)))
        os.remove(tmp_raw_fname)


def test_info_no_rename_no_reorder_no_pdf():
    """Test private renaming, reordering and partial construction option."""
    for pdf, config, hs in zip(pdf_fnames, config_fnames, hs_fnames):
        info, bti_info = _get_bti_info(
            pdf_fname=pdf, config_fname=config, head_shape_fname=hs,
            rotation_x=0.0, translation=(0.0, 0.02, 0.11), convert=False,
            ecg_ch='E31', eog_ch=('E63', 'E64'),
            rename_channels=False, sort_by_ch_name=False)
        info2, bti_info = _get_bti_info(
            pdf_fname=None, config_fname=config, head_shape_fname=hs,
            rotation_x=0.0, translation=(0.0, 0.02, 0.11), convert=False,
            ecg_ch='E31', eog_ch=('E63', 'E64'),
            rename_channels=False, sort_by_ch_name=False)

        assert_equal(info['ch_names'],
                     [ch['ch_name'] for ch in info['chs']])
        assert_equal([n for n in info['ch_names'] if n.startswith('A')][:5],
                     ['A22', 'A2', 'A104', 'A241', 'A138'])
        assert_equal([n for n in info['ch_names'] if n.startswith('A')][-5:],
                     ['A133', 'A158', 'A44', 'A134', 'A216'])

        info = pick_info(info, pick_types(info, meg=True, stim=True,
                                          resp=True))
        info2 = pick_info(info2, pick_types(info2, meg=True, stim=True,
                                            resp=True))

        assert (info['sfreq'] is not None)
        assert (info['lowpass'] is not None)
        assert (info['highpass'] is not None)
        assert (info['meas_date'] is not None)

        assert_equal(info2['sfreq'], None)
        assert_equal(info2['lowpass'], None)
        assert_equal(info2['highpass'], None)
        assert_equal(info2['meas_date'], None)

        assert_equal(info['ch_names'], info2['ch_names'])
        assert_equal(info['ch_names'], info2['ch_names'])
        for key in ['dev_ctf_t', 'dev_head_t', 'ctf_head_t']:
            assert_array_equal(info[key]['trans'], info2[key]['trans'])

        assert_array_equal(
            np.array([ch['loc'] for ch in info['chs']]),
            np.array([ch['loc'] for ch in info2['chs']]))

    # just check reading data | corner case
    raw1 = read_raw_bti(
        pdf_fname=pdf, config_fname=config, head_shape_fname=None,
        sort_by_ch_name=False, preload=True)
    # just check reading data | corner case
    raw2 = read_raw_bti(
        pdf_fname=pdf, config_fname=config, head_shape_fname=None,
        rename_channels=False,
        sort_by_ch_name=True, preload=True)

    sort_idx = [raw1.bti_ch_labels.index(ch) for ch in raw2.bti_ch_labels]
    raw1._data = raw1._data[sort_idx]
    assert_array_equal(raw1._data, raw2._data)
    assert_array_equal(raw2.bti_ch_labels, raw2.ch_names)


def test_no_conversion():
    """Test bti no-conversion option."""
    get_info = partial(
        _get_bti_info,
        rotation_x=0.0, translation=(0.0, 0.02, 0.11), convert=False,
        ecg_ch='E31', eog_ch=('E63', 'E64'),
        rename_channels=False, sort_by_ch_name=False)

    for pdf, config, hs in zip(pdf_fnames, config_fnames, hs_fnames):
        raw_info, _ = get_info(pdf, config, hs, convert=False)
        raw_info_con = read_raw_bti(
            pdf_fname=pdf, config_fname=config, head_shape_fname=hs,
            convert=True, preload=False).info

        pick_info(raw_info_con,
                  pick_types(raw_info_con, meg=True, ref_meg=True),
                  copy=False)
        pick_info(raw_info,
                  pick_types(raw_info, meg=True, ref_meg=True), copy=False)
        bti_info = _read_bti_header(pdf, config)
        dev_ctf_t = _correct_trans(bti_info['bti_transform'][0])
        assert_array_equal(dev_ctf_t, raw_info['dev_ctf_t']['trans'])
        assert_array_equal(raw_info['dev_head_t']['trans'], np.eye(4))
        assert_array_equal(raw_info['ctf_head_t']['trans'], np.eye(4))

        nasion, lpa, rpa, hpi, dig_points = _read_head_shape(hs)
        dig, t, _ = _make_bti_dig_points(nasion, lpa, rpa, hpi, dig_points,
                                         convert=False, use_hpi=False)

        assert_array_equal(t['trans'], np.eye(4))

        for ii, (old, new, con) in enumerate(zip(
                dig, raw_info['dig'], raw_info_con['dig'])):
            assert_equal(old['ident'], new['ident'])
            assert_array_equal(old['r'], new['r'])
            assert (not np.allclose(old['r'], con['r']))

            if ii > 10:
                break

        ch_map = {ch['chan_label']: ch['loc'] for ch in bti_info['chs']}

        for ii, ch_label in enumerate(raw_info['ch_names']):
            if not ch_label.startswith('A'):
                continue
            t1 = ch_map[ch_label]  # correction already performed in bti_info
            t2 = raw_info['chs'][ii]['loc']
            t3 = raw_info_con['chs'][ii]['loc']
            assert_allclose(t1, t2, atol=1e-15)
            assert (not np.allclose(t1, t3))
            idx_a = raw_info_con['ch_names'].index('MEG 001')
            idx_b = raw_info['ch_names'].index('A22')
            assert_equal(
                raw_info_con['chs'][idx_a]['coord_frame'],
                FIFF.FIFFV_COORD_DEVICE)
            assert_equal(
                raw_info['chs'][idx_b]['coord_frame'],
                FIFF.FIFFV_MNE_COORD_4D_HEAD)


def test_bytes_io():
    """Test bti bytes-io API."""
    for pdf, config, hs in zip(pdf_fnames, config_fnames, hs_fnames):
        raw = read_raw_bti(pdf, config, hs, convert=True, preload=False)

        with open(pdf, 'rb') as fid:
            pdf = BytesIO(fid.read())
        with open(config, 'rb') as fid:
            config = BytesIO(fid.read())
        with open(hs, 'rb') as fid:
            hs = BytesIO(fid.read())

        raw2 = read_raw_bti(pdf, config, hs, convert=True, preload=False)
        repr(raw2)
        assert_array_equal(raw[:][0], raw2[:][0])


def test_setup_headshape():
    """Test reading bti headshape."""
    for hs in hs_fnames:
        nasion, lpa, rpa, hpi, dig_points = _read_head_shape(hs)
        dig, t, _ = _make_bti_dig_points(nasion, lpa, rpa, hpi, dig_points)

        expected = {'kind', 'ident', 'r'}
        found = set(reduce(lambda x, y: list(x) + list(y),
                           [d.keys() for d in dig]))
        assert (not expected - found)


def test_nan_trans():
    """Test unlikely case that the device to head transform is empty."""
    for ii, pdf_fname in enumerate(pdf_fnames):
        bti_info = _read_bti_header(
            pdf_fname, config_fnames[ii], sort_by_ch_name=True)

        dev_ctf_t = Transform('ctf_meg', 'ctf_head',
                              _correct_trans(bti_info['bti_transform'][0]))

        # reading params
        convert = True
        rotation_x = 0.
        translation = (0.0, 0.02, 0.11)
        bti_dev_t = _get_bti_dev_t(rotation_x, translation)
        bti_dev_t = Transform('ctf_meg', 'meg', bti_dev_t)
        ecg_ch = 'E31'
        eog_ch = ('E63', 'E64')

        # read parts of info to get trans
        bti_ch_names = list()
        for ch in bti_info['chs']:
            ch_name = ch['name']
            if not ch_name.startswith('A'):
                ch_name = ch.get('chan_label', ch_name)
            bti_ch_names.append(ch_name)

        neuromag_ch_names = _rename_channels(
            bti_ch_names, ecg_ch=ecg_ch, eog_ch=eog_ch)
        ch_mapping = zip(bti_ch_names, neuromag_ch_names)

        # add some nan in some locations!
        dev_ctf_t['trans'][:, 3] = np.nan
        _check_nan_dev_head_t(dev_ctf_t)
        for idx, (chan_4d, chan_neuromag) in enumerate(ch_mapping):
            loc = bti_info['chs'][idx]['loc']
            if loc is not None:
                if convert:
                    t = _loc_to_coil_trans(bti_info['chs'][idx]['loc'])
                    t = _convert_coil_trans(t, dev_ctf_t, bti_dev_t)


@testing.requires_testing_data
@pytest.mark.parametrize('fname', (fname_sim, fname_sim_filt))
@pytest.mark.parametrize('preload', (True, False))
def test_bti_ch_data(fname, preload):
    """Test for gh-6048."""
    read_raw_bti(fname, preload=preload)  # used to fail with ascii decode err


run_tests_if_main()
