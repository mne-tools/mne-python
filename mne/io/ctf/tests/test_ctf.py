# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import copy
from datetime import datetime, timezone
import os
from os import path as op
import shutil

import numpy as np
from numpy import array_equal
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import mne
import mne.io.ctf.info
from mne import (pick_types, read_annotations, create_info,
                 events_from_annotations, make_forward_solution)
from mne.transforms import apply_trans
from mne.io import read_raw_fif, read_raw_ctf, RawArray
from mne.io.compensator import get_current_comp
from mne.io.ctf.constants import CTF
from mne.io.ctf.info import _convert_time
from mne.io.tests.test_raw import _test_raw_reader
from mne.tests.test_annotations import _assert_annotations_equal
from mne.utils import (_clean_names, catch_logging, _stamp_to_dt,
                       _record_warnings)
from mne.datasets import testing, spm_face, brainstorm
from mne.io.constants import FIFF

ctf_dir = testing.data_path(download=False) / 'CTF'
ctf_fname_continuous = 'testdata_ctf.ds'
ctf_fname_1_trial = 'testdata_ctf_short.ds'
ctf_fname_2_trials = 'testdata_ctf_pseudocontinuous.ds'
ctf_fname_discont = 'testdata_ctf_short_discontinuous.ds'
ctf_fname_somato = 'somMDYO-18av.ds'
ctf_fname_catch = 'catch-alp-good-f.ds'
somato_fname = op.join(
    brainstorm.bst_raw.data_path(download=False), 'MEG', 'bst_raw',
    'subj001_somatosensory_20111109_01_AUX-f.ds'
)
spm_path = spm_face.data_path(download=False)

block_sizes = {
    ctf_fname_continuous: 12000,
    ctf_fname_1_trial: 4801,
    ctf_fname_2_trials: 12000,
    ctf_fname_discont: 1201,
    ctf_fname_somato: 313,
    ctf_fname_catch: 2500,
}
single_trials = (
    ctf_fname_continuous,
    ctf_fname_1_trial,
)

ctf_fnames = tuple(sorted(block_sizes.keys()))


@pytest.mark.slowtest
@testing.requires_testing_data
def test_read_ctf(tmp_path):
    """Test CTF reader."""
    temp_dir = str(tmp_path)
    out_fname = op.join(temp_dir, 'test_py_raw.fif')

    # Create a dummy .eeg file so we can test our reading/application of it
    os.mkdir(op.join(temp_dir, 'randpos'))
    ctf_eeg_fname = op.join(temp_dir, 'randpos', ctf_fname_catch)
    shutil.copytree(op.join(ctf_dir, ctf_fname_catch), ctf_eeg_fname)
    with pytest.warns(RuntimeWarning, match='RMSP .* changed to a MISC ch'):
        raw = _test_raw_reader(read_raw_ctf, directory=ctf_eeg_fname)
    picks = pick_types(raw.info, meg=False, eeg=True)
    pos = np.random.RandomState(42).randn(len(picks), 3)
    fake_eeg_fname = op.join(ctf_eeg_fname, 'catch-alp-good-f.eeg')
    # Create a bad file
    with open(fake_eeg_fname, 'wb') as fid:
        fid.write('foo\n'.encode('ascii'))
    pytest.raises(RuntimeError, read_raw_ctf, ctf_eeg_fname)
    # Create a good file
    with open(fake_eeg_fname, 'wb') as fid:
        for ii, ch_num in enumerate(picks):
            args = (str(ch_num + 1), raw.ch_names[ch_num],) + tuple(
                '%0.5f' % x for x in 100 * pos[ii])  # convert to cm
            fid.write(('\t'.join(args) + '\n').encode('ascii'))
    pos_read_old = np.array([raw.info['chs'][p]['loc'][:3] for p in picks])
    with pytest.warns(RuntimeWarning, match='RMSP .* changed to a MISC ch'):
        raw = read_raw_ctf(ctf_eeg_fname)  # read modified data
    pos_read = np.array([raw.info['chs'][p]['loc'][:3] for p in picks])
    assert_allclose(apply_trans(raw.info['ctf_head_t'], pos), pos_read,
                    rtol=1e-5, atol=1e-5)
    assert (pos_read == pos_read_old).mean() < 0.1
    shutil.copy(op.join(ctf_dir, 'catch-alp-good-f.ds_randpos_raw.fif'),
                op.join(temp_dir, 'randpos', 'catch-alp-good-f.ds_raw.fif'))

    # Create a version with no hc, starting out *with* EEG pos (error)
    os.mkdir(op.join(temp_dir, 'nohc'))
    ctf_no_hc_fname = op.join(temp_dir, 'no_hc', ctf_fname_catch)
    shutil.copytree(ctf_eeg_fname, ctf_no_hc_fname)
    remove_base = op.join(ctf_no_hc_fname, op.basename(ctf_fname_catch[:-3]))
    os.remove(remove_base + '.hc')
    with pytest.warns(RuntimeWarning, match='MISC channel'):
        pytest.raises(RuntimeError, read_raw_ctf, ctf_no_hc_fname)
    os.remove(remove_base + '.eeg')
    shutil.copy(op.join(ctf_dir, 'catch-alp-good-f.ds_nohc_raw.fif'),
                op.join(temp_dir, 'no_hc', 'catch-alp-good-f.ds_raw.fif'))

    # All our files
    use_fnames = [op.join(ctf_dir, c) for c in ctf_fnames]
    for fname in use_fnames:
        raw_c = read_raw_fif(fname + '_raw.fif', preload=True)
        # sometimes matches "MISC channel"
        with _record_warnings():
            raw = read_raw_ctf(fname)

        # check info match
        assert_array_equal(raw.ch_names, raw_c.ch_names)
        assert_allclose(raw.times, raw_c.times)
        assert_allclose(raw._cals, raw_c._cals)
        assert (raw.info['meas_id']['version'] ==
                raw_c.info['meas_id']['version'] + 1)
        for t in ('dev_head_t', 'dev_ctf_t', 'ctf_head_t'):
            assert_allclose(raw.info[t]['trans'], raw_c.info[t]['trans'],
                            rtol=1e-4, atol=1e-7)
        # XXX 2019/11/29 : MNC-C FIF conversion files don't have meas_date set.
        # Consider adding meas_date to below checks once this is addressed in
        # MNE-C
        for key in ('acq_pars', 'acq_stim', 'bads',
                    'ch_names', 'custom_ref_applied', 'description',
                    'events', 'experimenter', 'highpass', 'line_freq',
                    'lowpass', 'nchan', 'proj_id', 'proj_name',
                    'projs', 'sfreq', 'subject_info'):
            assert raw.info[key] == raw_c.info[key], key
        if op.basename(fname) not in single_trials:
            # We don't force buffer size to be smaller like MNE-C
            assert raw.buffer_size_sec == raw_c.buffer_size_sec
        assert len(raw.info['comps']) == len(raw_c.info['comps'])
        for c1, c2 in zip(raw.info['comps'], raw_c.info['comps']):
            for key in ('colcals', 'rowcals'):
                assert_allclose(c1[key], c2[key])
            assert c1['save_calibrated'] == c2['save_calibrated']
            for key in ('row_names', 'col_names', 'nrow', 'ncol'):
                assert_array_equal(c1['data'][key], c2['data'][key])
            assert_allclose(c1['data']['data'], c2['data']['data'], atol=1e-7,
                            rtol=1e-5)
        assert_allclose(raw.info['hpi_results'][0]['coord_trans']['trans'],
                        raw_c.info['hpi_results'][0]['coord_trans']['trans'],
                        rtol=1e-5, atol=1e-7)
        assert len(raw.info['chs']) == len(raw_c.info['chs'])
        for ii, (c1, c2) in enumerate(zip(raw.info['chs'], raw_c.info['chs'])):
            for key in ('kind', 'scanno', 'unit', 'ch_name', 'unit_mul',
                        'range', 'coord_frame', 'coil_type', 'logno'):
                if c1['ch_name'] == 'RMSP' and \
                        'catch-alp-good-f' in fname and \
                        key in ('kind', 'unit', 'coord_frame', 'coil_type',
                                'logno'):
                    continue  # XXX see below...
                if key == 'coil_type' and c1[key] == FIFF.FIFFV_COIL_EEG:
                    # XXX MNE-C bug that this is not set
                    assert c2[key] == FIFF.FIFFV_COIL_NONE
                    continue
                assert c1[key] == c2[key], key
            for key in ('cal',):
                assert_allclose(c1[key], c2[key], atol=1e-6, rtol=1e-4,
                                err_msg='raw.info["chs"][%d][%s]' % (ii, key))
            # XXX 2016/02/24: fixed bug with normal computation that used
            # to exist, once mne-C tools are updated we should update our FIF
            # conversion files, then the slices can go away (and the check
            # can be combined with that for "cal")
            for key in ('loc',):
                if c1['ch_name'] == 'RMSP' and 'catch-alp-good-f' in fname:
                    continue
                if (c2[key][:3] == 0.).all():
                    check = [np.nan] * 3
                else:
                    check = c2[key][:3]
                assert_allclose(c1[key][:3], check, atol=1e-6, rtol=1e-4,
                                err_msg='raw.info["chs"][%d][%s]' % (ii, key))
                if (c2[key][3:] == 0.).all():
                    check = [np.nan] * 3
                else:
                    check = c2[key][9:12]
                assert_allclose(c1[key][9:12], check, atol=1e-6, rtol=1e-4,
                                err_msg='raw.info["chs"][%d][%s]' % (ii, key))

        # Make sure all digitization points are in the MNE head coord frame
        for p in raw.info['dig']:
            assert p['coord_frame'] == FIFF.FIFFV_COORD_HEAD, \
                'dig points must be in FIFF.FIFFV_COORD_HEAD'

        if fname.endswith('catch-alp-good-f.ds'):  # omit points from .pos file
            with raw.info._unlock():
                raw.info['dig'] = raw.info['dig'][:-10]

        # XXX: Next test would fail because c-tools assign the fiducials from
        # CTF data as HPI. Should eventually clarify/unify with Matti.
        # assert_dig_allclose(raw.info, raw_c.info)

        # check data match
        raw_c.save(out_fname, overwrite=True, buffer_size_sec=1.)
        raw_read = read_raw_fif(out_fname)

        # so let's check tricky cases based on sample boundaries
        rng = np.random.RandomState(0)
        pick_ch = rng.permutation(np.arange(len(raw.ch_names)))[:10]
        bnd = int(round(raw.info['sfreq'] * raw.buffer_size_sec))
        assert bnd == raw._raw_extras[0]['block_size']
        assert bnd == block_sizes[op.basename(fname)]
        slices = (slice(0, bnd), slice(bnd - 1, bnd), slice(3, bnd),
                  slice(3, 300), slice(None))
        if len(raw.times) >= 2 * bnd:  # at least two complete blocks
            slices = slices + (slice(bnd, 2 * bnd), slice(bnd, bnd + 1),
                               slice(0, bnd + 100))
        for sl_time in slices:
            assert_allclose(raw[pick_ch, sl_time][0],
                            raw_c[pick_ch, sl_time][0])
            assert_allclose(raw_read[pick_ch, sl_time][0],
                            raw_c[pick_ch, sl_time][0])
        # all data / preload
        raw.load_data()
        assert_allclose(raw[:][0], raw_c[:][0], atol=1e-15)
        # test bad segment annotations
        if 'testdata_ctf_short.ds' in fname:
            assert 'bad' in raw.annotations.description[0]
            assert_allclose(raw.annotations.onset, [2.15])
            assert_allclose(raw.annotations.duration, [0.0225])

    with pytest.raises(TypeError, match='path-like'):
        read_raw_ctf(1)
    with pytest.raises(FileNotFoundError, match='does not exist'):
        read_raw_ctf(ctf_fname_continuous + 'foo.ds')
    # test ignoring of system clock
    read_raw_ctf(op.join(ctf_dir, ctf_fname_continuous), 'ignore')
    with pytest.raises(ValueError, match='system_clock'):
        read_raw_ctf(op.join(ctf_dir, ctf_fname_continuous), 'foo')


@testing.requires_testing_data
def test_rawctf_clean_names():
    """Test RawCTF _clean_names method."""
    # read test data
    with pytest.warns(RuntimeWarning, match='ref channel RMSP did not'):
        raw = read_raw_ctf(op.join(ctf_dir, ctf_fname_catch))
        raw_cleaned = read_raw_ctf(op.join(ctf_dir, ctf_fname_catch),
                                   clean_names=True)
    test_channel_names = _clean_names(raw.ch_names)
    test_info_comps = copy.deepcopy(raw.info['comps'])

    # channel names should not be cleaned by default
    assert raw.ch_names != test_channel_names

    chs_ch_names = [ch['ch_name'] for ch in raw.info['chs']]

    assert chs_ch_names != test_channel_names

    for test_comp, comp in zip(test_info_comps, raw.info['comps']):
        for key in ('row_names', 'col_names'):
            assert not array_equal(_clean_names(test_comp['data'][key]),
                                   comp['data'][key])

    # channel names should be cleaned if clean_names=True
    assert raw_cleaned.ch_names == test_channel_names

    for ch, test_ch_name in zip(raw_cleaned.info['chs'], test_channel_names):
        assert ch['ch_name'] == test_ch_name

    for test_comp, comp in zip(test_info_comps, raw_cleaned.info['comps']):
        for key in ('row_names', 'col_names'):
            assert _clean_names(test_comp['data'][key]) == comp['data'][key]


@spm_face.requires_spm_data
def test_read_spm_ctf():
    """Test CTF reader with omitted samples."""
    raw_fname = op.join(spm_path, 'MEG', 'spm',
                        'SPM_CTF_MEG_example_faces1_3D.ds')
    raw = read_raw_ctf(raw_fname)
    extras = raw._raw_extras[0]
    assert extras['n_samp'] == raw.n_times
    assert extras['n_samp'] != extras['n_samp_tot']

    # Test that LPA, nasion and RPA are correct.
    coord_frames = np.array([d['coord_frame'] for d in raw.info['dig']])
    assert np.all(coord_frames == FIFF.FIFFV_COORD_HEAD)
    cardinals = {d['ident']: d['r'] for d in raw.info['dig']}
    assert cardinals[1][0] < cardinals[2][0] < cardinals[3][0]  # x coord
    assert cardinals[1][1] < cardinals[2][1]  # y coord
    assert cardinals[3][1] < cardinals[2][1]  # y coord
    for key in cardinals.keys():
        assert_allclose(cardinals[key][2], 0, atol=1e-6)  # z coord


@testing.requires_testing_data
@pytest.mark.parametrize('comp_grade', [0, 1])
def test_saving_picked(tmp_path, comp_grade):
    """Test saving picked CTF instances."""
    temp_dir = str(tmp_path)
    out_fname = op.join(temp_dir, 'test_py_raw.fif')
    raw = read_raw_ctf(op.join(ctf_dir, ctf_fname_1_trial))
    assert raw.info['meas_date'] == _stamp_to_dt((1367228160, 0))
    raw.crop(0, 1).load_data()
    assert raw.compensation_grade == get_current_comp(raw.info) == 0
    assert len(raw.info['comps']) == 5
    pick_kwargs = dict(meg=True, ref_meg=False, verbose=True)

    raw.apply_gradient_compensation(comp_grade)
    with catch_logging() as log:
        raw_pick = raw.copy().pick_types(**pick_kwargs)
    assert len(raw.info['comps']) == 5
    assert len(raw_pick.info['comps']) == 0
    log = log.getvalue()
    assert 'Removing 5 compensators' in log
    raw_pick.save(out_fname, overwrite=True)  # should work
    raw2 = read_raw_fif(out_fname)
    assert (raw_pick.ch_names == raw2.ch_names)
    assert_array_equal(raw_pick.times, raw2.times)
    assert_allclose(raw2[0:20][0], raw_pick[0:20][0], rtol=1e-6,
                    atol=1e-20)  # atol is very small but > 0

    raw2 = read_raw_fif(out_fname, preload=True)
    assert (raw_pick.ch_names == raw2.ch_names)
    assert_array_equal(raw_pick.times, raw2.times)
    assert_allclose(raw2[0:20][0], raw_pick[0:20][0], rtol=1e-6,
                    atol=1e-20)  # atol is very small but > 0


@brainstorm.bst_raw.requires_bstraw_data
def test_read_ctf_annotations():
    """Test reading CTF marker file."""
    EXPECTED_LATENCIES = np.array([
         5640,   7950,   9990,  12253,  14171,  16557,  18896,  20846,  # noqa
        22702,  24990,  26830,  28974,  30906,  33077,  34985,  36907,  # noqa
        38922,  40760,  42881,  45222,  47457,  49618,  51802,  54227,  # noqa
        56171,  58274,  60394,  62375,  64444,  66767,  68827,  71109,  # noqa
        73499,  75807,  78146,  80415,  82554,  84508,  86403,  88426,  # noqa
        90746,  92893,  94779,  96822,  98996,  99001, 100949, 103325,  # noqa
       105322, 107678, 109667, 111844, 113682, 115817, 117691, 119663,  # noqa
       121966, 123831, 126110, 128490, 130521, 132808, 135204, 137210,  # noqa
       139130, 141390, 143660, 145748, 147889, 150205, 152528, 154646,  # noqa
       156897, 159191, 161446, 163722, 166077, 168467, 170624, 172519,  # noqa
       174719, 176886, 179062, 181405, 183709, 186034, 188454, 190330,  # noqa
       192660, 194682, 196834, 199161, 201035, 203008, 204999, 207409,  # noqa
       209661, 211895, 213957, 216005, 218040, 220178, 222137, 224305,  # noqa
       226297, 228654, 230755, 232909, 235205, 237373, 239723, 241762,  # noqa
       243748, 245762, 247801, 250055, 251886, 254252, 256441, 258354,  # noqa
       260680, 263026, 265048, 267073, 269235, 271556, 273927, 276197,  # noqa
       278436, 280536, 282691, 284933, 287061, 288936, 290941, 293183,  # noqa
       295369, 297729, 299626, 301546, 303449, 305548, 307882, 310124,  # noqa
       312374, 314509, 316815, 318789, 320981, 322879, 324878, 326959,  # noqa
       329341, 331200, 331201, 333469, 335584, 337984, 340143, 342034,  # noqa
       344360, 346309, 348544, 350970, 353052, 355227, 357449, 359603,  # noqa
       361725, 363676, 365735, 367799, 369777, 371904, 373856, 376204,  # noqa
       378391, 380800, 382859, 385161, 387093, 389434, 391624, 393785,  # noqa
       396093, 398214, 400198, 402166, 404104, 406047, 408372, 410686,  # noqa
       413029, 414975, 416850, 418797, 420824, 422959, 425026, 427215,  # noqa
       429278, 431668  # noqa
    ]) - 1  # Fieldtrip has 1 sample difference with MNE

    raw = RawArray(
        data=np.empty((1, 432000), dtype=np.float64),
        info=create_info(ch_names=1, sfreq=1200.0))
    raw.set_meas_date(read_raw_ctf(somato_fname).info['meas_date'])
    raw.set_annotations(read_annotations(somato_fname))

    events, _ = events_from_annotations(raw)
    latencies = np.sort(events[:, 0])
    assert_allclose(latencies, EXPECTED_LATENCIES, atol=1e-6)


@testing.requires_testing_data
def test_read_ctf_annotations_smoke_test():
    """Test reading CTF marker file.

    `testdata_ctf_mc.ds` has no trials or offsets therefore its a plain reading
    of whatever is in the MarkerFile.mrk.
    """
    EXPECTED_ONSET = [
        0., 0.1425, 0.285, 0.42833333, 0.57083333, 0.71416667, 0.85666667,
        0.99916667, 1.1425, 1.285, 1.4275, 1.57083333, 1.71333333, 1.85666667,
        1.99916667, 2.14166667, 2.285, 2.4275, 2.57083333, 2.71333333,
        2.85583333, 2.99916667, 3.14166667, 3.28416667, 3.4275, 3.57,
        3.71333333, 3.85583333, 3.99833333, 4.14166667, 4.28416667, 4.42666667,
        4.57, 4.7125, 4.85583333, 4.99833333
    ]
    fname = op.join(ctf_dir, 'testdata_ctf_mc.ds')
    annot = read_annotations(fname)
    assert_allclose(annot.onset, EXPECTED_ONSET)

    raw = read_raw_ctf(fname)
    _assert_annotations_equal(raw.annotations, annot, 1e-6)


def _read_res4_mag_comp(dsdir):
    res = mne.io.ctf.res4._read_res4(dsdir)
    for ch in res['chs']:
        if ch['sensor_type_index'] == CTF.CTFV_REF_MAG_CH:
            ch['grad_order_no'] = 1
    return res


def _bad_res4_grad_comp(dsdir):
    res = mne.io.ctf.res4._read_res4(dsdir)
    for ch in res['chs']:
        if ch['sensor_type_index'] == CTF.CTFV_MEG_CH:
            ch['grad_order_no'] = 1
            break
    return res


@testing.requires_testing_data
def test_missing_res4(tmp_path):
    """Test that res4 missing is handled gracefully."""
    use_ds = tmp_path / ctf_fname_continuous
    shutil.copytree(ctf_dir / ctf_fname_continuous,
                    tmp_path / ctf_fname_continuous)
    read_raw_ctf(use_ds)
    os.remove(use_ds / (ctf_fname_continuous[:-2] + 'meg4'))
    with pytest.raises(IOError, match='could not find the following'):
        read_raw_ctf(use_ds)


@testing.requires_testing_data
def test_read_ctf_mag_bad_comp(tmp_path, monkeypatch):
    """Test CTF reader with mag comps and bad comps."""
    path = op.join(ctf_dir, ctf_fname_continuous)
    raw_orig = read_raw_ctf(path)
    assert raw_orig.compensation_grade == 0
    monkeypatch.setattr(mne.io.ctf.ctf, '_read_res4', _read_res4_mag_comp)
    raw_mag_comp = read_raw_ctf(path)
    assert raw_mag_comp.compensation_grade == 0
    sphere = mne.make_sphere_model()
    src = mne.setup_volume_source_space(pos=50., exclude=5., bem=sphere)
    assert src[0]['nuse'] == 26
    for grade in (0, 1):
        raw_orig.apply_gradient_compensation(grade)
        raw_mag_comp.apply_gradient_compensation(grade)
        args = (None, src, sphere, True, False)
        fwd_orig = make_forward_solution(raw_orig.info, *args)
        fwd_mag_comp = make_forward_solution(raw_mag_comp.info, *args)
        assert_allclose(fwd_orig['sol']['data'], fwd_mag_comp['sol']['data'])
    monkeypatch.setattr(mne.io.ctf.ctf, '_read_res4', _bad_res4_grad_comp)
    with pytest.raises(RuntimeError, match='inconsistent compensation grade'):
        read_raw_ctf(path)


@testing.requires_testing_data
def test_invalid_meas_date(monkeypatch):
    """Test handling of invalid meas_date."""
    def _convert_time_bad(date_str, time_str):
        return _convert_time('', '')
    monkeypatch.setattr(mne.io.ctf.info, '_convert_time', _convert_time_bad)

    with catch_logging() as log:
        raw = read_raw_ctf(ctf_dir / ctf_fname_continuous, verbose=True)
    log = log.getvalue()
    assert 'No date or time found' in log
    assert raw.info['meas_date'] == datetime.fromtimestamp(0, tz=timezone.utc)
