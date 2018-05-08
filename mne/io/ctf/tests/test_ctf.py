# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import copy
import os
from os import path as op
import shutil
import warnings

import numpy as np
from numpy import array_equal
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
import pytest

from mne import pick_types
# from mne.tests.common import assert_dig_allclose
from mne.transforms import apply_trans
from mne.io import read_raw_fif, read_raw_ctf
from mne.io.compensator import get_current_comp
from mne.io.tests.test_raw import _test_raw_reader
from mne.utils import _TempDir, run_tests_if_main, _clean_names, catch_logging
from mne.datasets import testing, spm_face
from mne.io.constants import FIFF

ctf_dir = op.join(testing.data_path(download=False), 'CTF')
ctf_fname_continuous = 'testdata_ctf.ds'
ctf_fname_1_trial = 'testdata_ctf_short.ds'
ctf_fname_2_trials = 'testdata_ctf_pseudocontinuous.ds'
ctf_fname_discont = 'testdata_ctf_short_discontinuous.ds'
ctf_fname_somato = 'somMDYO-18av.ds'
ctf_fname_catch = 'catch-alp-good-f.ds'

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
def test_read_ctf():
    """Test CTF reader"""
    temp_dir = _TempDir()
    out_fname = op.join(temp_dir, 'test_py_raw.fif')

    # Create a dummy .eeg file so we can test our reading/application of it
    os.mkdir(op.join(temp_dir, 'randpos'))
    ctf_eeg_fname = op.join(temp_dir, 'randpos', ctf_fname_catch)
    shutil.copytree(op.join(ctf_dir, ctf_fname_catch), ctf_eeg_fname)
    with warnings.catch_warnings(record=True) as w:  # reclassified ch
        raw = _test_raw_reader(read_raw_ctf, directory=ctf_eeg_fname)
    assert all('MISC channel' in str(ww.message) for ww in w)
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
    with warnings.catch_warnings(record=True) as w:  # reclassified channel
        raw = read_raw_ctf(ctf_eeg_fname)  # read modified data
    assert all('MISC channel' in str(ww.message) for ww in w)
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
    with warnings.catch_warnings(record=True):  # no coord tr
        pytest.raises(RuntimeError, read_raw_ctf, ctf_no_hc_fname)
    os.remove(remove_base + '.eeg')
    shutil.copy(op.join(ctf_dir, 'catch-alp-good-f.ds_nohc_raw.fif'),
                op.join(temp_dir, 'no_hc', 'catch-alp-good-f.ds_raw.fif'))

    # All our files
    use_fnames = [op.join(ctf_dir, c) for c in ctf_fnames]
    for fname in use_fnames:
        raw_c = read_raw_fif(fname + '_raw.fif', preload=True)
        with warnings.catch_warnings(record=True) as w:  # reclassified ch
            raw = read_raw_ctf(fname)
        assert all('MISC channel' in str(ww.message) for ww in w)

        # check info match
        assert_array_equal(raw.ch_names, raw_c.ch_names)
        assert_allclose(raw.times, raw_c.times)
        assert_allclose(raw._cals, raw_c._cals)
        for key in ('version', 'usecs'):
            assert_equal(raw.info['meas_id'][key], raw_c.info['meas_id'][key])
        py_time = raw.info['meas_id']['secs']
        c_time = raw_c.info['meas_id']['secs']
        max_offset = 24 * 60 * 60  # probably overkill but covers timezone
        assert c_time - max_offset <= py_time <= c_time
        for t in ('dev_head_t', 'dev_ctf_t', 'ctf_head_t'):
            assert_allclose(raw.info[t]['trans'], raw_c.info[t]['trans'],
                            rtol=1e-4, atol=1e-7)
        for key in ('acq_pars', 'acq_stim', 'bads',
                    'ch_names', 'custom_ref_applied', 'description',
                    'events', 'experimenter', 'highpass', 'line_freq',
                    'lowpass', 'nchan', 'proj_id', 'proj_name',
                    'projs', 'sfreq', 'subject_info'):
            assert_equal(raw.info[key], raw_c.info[key], key)
        if op.basename(fname) not in single_trials:
            # We don't force buffer size to be smaller like MNE-C
            assert_equal(raw.info['buffer_size_sec'],
                         raw_c.info['buffer_size_sec'])
        assert_equal(len(raw.info['comps']), len(raw_c.info['comps']))
        for c1, c2 in zip(raw.info['comps'], raw_c.info['comps']):
            for key in ('colcals', 'rowcals'):
                assert_allclose(c1[key], c2[key])
            assert_equal(c1['save_calibrated'], c2['save_calibrated'])
            for key in ('row_names', 'col_names', 'nrow', 'ncol'):
                assert_array_equal(c1['data'][key], c2['data'][key])
            assert_allclose(c1['data']['data'], c2['data']['data'], atol=1e-7,
                            rtol=1e-5)
        assert_allclose(raw.info['hpi_results'][0]['coord_trans']['trans'],
                        raw_c.info['hpi_results'][0]['coord_trans']['trans'],
                        rtol=1e-5, atol=1e-7)
        assert_equal(len(raw.info['chs']), len(raw_c.info['chs']))
        for ii, (c1, c2) in enumerate(zip(raw.info['chs'], raw_c.info['chs'])):
            for key in ('kind', 'scanno', 'unit', 'ch_name', 'unit_mul',
                        'range', 'coord_frame', 'coil_type', 'logno'):
                if c1['ch_name'] == 'RMSP' and \
                        'catch-alp-good-f' in fname and \
                        key in ('kind', 'unit', 'coord_frame', 'coil_type',
                                'logno'):
                    continue  # XXX see below...
                assert_equal(c1[key], c2[key], err_msg=key)
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
            assert_equal(p['coord_frame'], FIFF.FIFFV_COORD_HEAD,
                         err_msg='dig points must be in FIFF.FIFFV_COORD_HEAD')

        if fname.endswith('catch-alp-good-f.ds'):  # omit points from .pos file
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
        bnd = int(round(raw.info['sfreq'] * raw.info['buffer_size_sec']))
        assert_equal(bnd, raw._raw_extras[0]['block_size'])
        assert_equal(bnd, block_sizes[op.basename(fname)])
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
        with warnings.catch_warnings(record=True) as w:  # reclassified ch
            raw = read_raw_ctf(fname, preload=True)
        assert all('MISC channel' in str(ww.message) for ww in w)
        assert_allclose(raw[:][0], raw_c[:][0])
    pytest.raises(TypeError, read_raw_ctf, 1)
    pytest.raises(ValueError, read_raw_ctf, ctf_fname_continuous + 'foo.ds')
    # test ignoring of system clock
    read_raw_ctf(op.join(ctf_dir, ctf_fname_continuous), 'ignore')
    pytest.raises(ValueError, read_raw_ctf,
                  op.join(ctf_dir, ctf_fname_continuous), 'foo')


@testing.requires_testing_data
def test_rawctf_clean_names():
    """Test RawCTF _clean_names method."""
    # read test data
    with warnings.catch_warnings(record=True) as w:
        raw = read_raw_ctf(op.join(ctf_dir, ctf_fname_catch))
        raw_cleaned = read_raw_ctf(op.join(ctf_dir, ctf_fname_catch),
                                   clean_names=True)
    assert any('MEG ref channel RMSP did not' in str(ww.message) for ww in w)
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
    data_path = spm_face.data_path()
    raw_fname = op.join(data_path, 'MEG', 'spm',
                        'SPM_CTF_MEG_example_faces1_3D.ds')
    raw = read_raw_ctf(raw_fname)
    extras = raw._raw_extras[0]
    assert_equal(extras['n_samp'], raw.n_times)
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
def test_saving_picked():
    """Test saving picked CTF instances."""
    temp_dir = _TempDir()
    out_fname = op.join(temp_dir, 'test_py_raw.fif')
    raw = read_raw_ctf(op.join(ctf_dir, ctf_fname_1_trial))
    raw.crop(0, 1).load_data()
    assert raw.compensation_grade == get_current_comp(raw.info) == 0
    assert len(raw.info['comps']) == 5
    pick_kwargs = dict(meg=True, ref_meg=False, verbose=True)
    with catch_logging() as log:
        raw_pick = raw.copy().pick_types(**pick_kwargs)
    assert len(raw.info['comps']) == 5
    assert len(raw_pick.info['comps']) == 0
    log = log.getvalue()
    assert 'Removing 5 compensators' in log
    raw_pick.save(out_fname)  # should work
    read_raw_fif(out_fname)
    read_raw_fif(out_fname, preload=True)
    # If comp is applied, picking should error
    raw.apply_gradient_compensation(1)
    assert raw.compensation_grade == get_current_comp(raw.info) == 1
    with pytest.raises(RuntimeError, match='Compensation grade 1 has been'):
        raw.copy().pick_types(**pick_kwargs)


run_tests_if_main()
