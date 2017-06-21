import inspect
import os.path as op
import warnings

from nose.tools import assert_equal, assert_raises, assert_true
from numpy.testing import assert_array_equal
import numpy as np

from mne import (pick_channels_regexp, pick_types, Epochs,
                 read_forward_solution, rename_channels,
                 pick_info, pick_channels, __file__, create_info)
from mne.io import (read_raw_fif, RawArray, read_raw_bti, read_raw_kit,
                    read_info)
from mne.io.pick import (channel_indices_by_type, channel_type,
                         pick_types_forward, _picks_by_type)
from mne.io.constants import FIFF
from mne.datasets import testing
from mne.utils import run_tests_if_main

io_dir = op.join(op.dirname(inspect.getfile(inspect.currentframe())), '..')
data_path = testing.data_path(download=False)
fname_meeg = op.join(data_path, 'MEG', 'sample',
                     'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
fname_mc = op.join(data_path, 'SSS', 'test_move_anon_movecomp_raw_sss.fif')


def test_pick_refs():
    """Test picking of reference sensors."""
    infos = list()
    # KIT
    kit_dir = op.join(io_dir, 'kit', 'tests', 'data')
    sqd_path = op.join(kit_dir, 'test.sqd')
    mrk_path = op.join(kit_dir, 'test_mrk.sqd')
    elp_path = op.join(kit_dir, 'test_elp.txt')
    hsp_path = op.join(kit_dir, 'test_hsp.txt')
    raw_kit = read_raw_kit(sqd_path, mrk_path, elp_path, hsp_path)
    infos.append(raw_kit.info)
    # BTi
    bti_dir = op.join(io_dir, 'bti', 'tests', 'data')
    bti_pdf = op.join(bti_dir, 'test_pdf_linux')
    bti_config = op.join(bti_dir, 'test_config_linux')
    bti_hs = op.join(bti_dir, 'test_hs_linux')
    with warnings.catch_warnings(record=True):  # weight tables
        raw_bti = read_raw_bti(bti_pdf, bti_config, bti_hs, preload=False)
    infos.append(raw_bti.info)
    # CTF
    fname_ctf_raw = op.join(io_dir, 'tests', 'data', 'test_ctf_comp_raw.fif')
    raw_ctf = read_raw_fif(fname_ctf_raw)
    raw_ctf.apply_gradient_compensation(2)
    infos.append(raw_ctf.info)
    for info in infos:
        info['bads'] = []
        assert_raises(ValueError, pick_types, info, meg='foo')
        assert_raises(ValueError, pick_types, info, ref_meg='foo')
        picks_meg_ref = pick_types(info, meg=True, ref_meg=True)
        picks_meg = pick_types(info, meg=True, ref_meg=False)
        picks_ref = pick_types(info, meg=False, ref_meg=True)
        assert_array_equal(picks_meg_ref,
                           np.sort(np.concatenate([picks_meg, picks_ref])))
        picks_grad = pick_types(info, meg='grad', ref_meg=False)
        picks_ref_grad = pick_types(info, meg=False, ref_meg='grad')
        picks_meg_ref_grad = pick_types(info, meg='grad', ref_meg='grad')
        assert_array_equal(picks_meg_ref_grad,
                           np.sort(np.concatenate([picks_grad,
                                                   picks_ref_grad])))
        picks_mag = pick_types(info, meg='mag', ref_meg=False)
        picks_ref_mag = pick_types(info, meg=False, ref_meg='mag')
        picks_meg_ref_mag = pick_types(info, meg='mag', ref_meg='mag')
        assert_array_equal(picks_meg_ref_mag,
                           np.sort(np.concatenate([picks_mag,
                                                   picks_ref_mag])))
        assert_array_equal(picks_meg,
                           np.sort(np.concatenate([picks_mag, picks_grad])))
        assert_array_equal(picks_ref,
                           np.sort(np.concatenate([picks_ref_mag,
                                                   picks_ref_grad])))
        assert_array_equal(picks_meg_ref, np.sort(np.concatenate(
            [picks_grad, picks_mag, picks_ref_grad, picks_ref_mag])))
        for pick in (picks_meg_ref, picks_meg, picks_ref,
                     picks_grad, picks_ref_grad, picks_meg_ref_grad,
                     picks_mag, picks_ref_mag, picks_meg_ref_mag):
            if len(pick) > 0:
                pick_info(info, pick)


def test_pick_channels_regexp():
    """Test pick with regular expression
    """
    ch_names = ['MEG 2331', 'MEG 2332', 'MEG 2333']
    assert_array_equal(pick_channels_regexp(ch_names, 'MEG ...1'), [0])
    assert_array_equal(pick_channels_regexp(ch_names, 'MEG ...[2-3]'), [1, 2])
    assert_array_equal(pick_channels_regexp(ch_names, 'MEG *'), [0, 1, 2])


def test_pick_seeg_ecog():
    """Test picking with sEEG and ECoG
    """
    names = 'A1 A2 Fz O OTp1 OTp2 E1 OTp3 E2 E3'.split()
    types = 'mag mag eeg eeg seeg seeg ecog seeg ecog ecog'.split()
    info = create_info(names, 1024., types)
    idx = channel_indices_by_type(info)
    assert_array_equal(idx['mag'], [0, 1])
    assert_array_equal(idx['eeg'], [2, 3])
    assert_array_equal(idx['seeg'], [4, 5, 7])
    assert_array_equal(idx['ecog'], [6, 8, 9])
    assert_array_equal(pick_types(info, meg=False, seeg=True), [4, 5, 7])
    for i, t in enumerate(types):
        assert_equal(channel_type(info, i), types[i])
    raw = RawArray(np.zeros((len(names), 10)), info)
    events = np.array([[1, 0, 0], [2, 0, 0]])
    epochs = Epochs(raw, events, {'event': 0}, -1e-5, 1e-5)
    evoked = epochs.average(pick_types(epochs.info, meg=True, seeg=True))
    e_seeg = evoked.copy().pick_types(meg=False, seeg=True)
    for l, r in zip(e_seeg.ch_names, [names[4], names[5], names[7]]):
        assert_equal(l, r)
    # Deal with constant debacle
    raw = read_raw_fif(op.join(io_dir, 'tests', 'data',
                               'test_chpi_raw_sss.fif'))
    assert_equal(len(pick_types(raw.info, meg=False, seeg=True, ecog=True)), 0)


def test_pick_chpi():
    """Test picking cHPI
    """
    # Make sure we don't mis-classify cHPI channels
    info = read_info(op.join(io_dir, 'tests', 'data', 'test_chpi_raw_sss.fif'))
    channel_types = set([channel_type(info, idx)
                         for idx in range(info['nchan'])])
    assert_true('chpi' in channel_types)
    assert_true('seeg' not in channel_types)
    assert_true('ecog' not in channel_types)


def test_pick_bio():
    """Test picking BIO channels."""
    names = 'A1 A2 Fz O BIO1 BIO2 BIO3'.split()
    types = 'mag mag eeg eeg bio bio bio'.split()
    info = create_info(names, 1024., types)
    idx = channel_indices_by_type(info)
    assert_array_equal(idx['mag'], [0, 1])
    assert_array_equal(idx['eeg'], [2, 3])
    assert_array_equal(idx['bio'], [4, 5, 6])


def test_pick_fnirs():
    """Test picking fNIRS channels."""
    names = 'A1 A2 Fz O hbo1 hbo2 hbr1'.split()
    types = 'mag mag eeg eeg hbo hbo hbr'.split()
    info = create_info(names, 1024., types)
    idx = channel_indices_by_type(info)
    assert_array_equal(idx['mag'], [0, 1])
    assert_array_equal(idx['eeg'], [2, 3])
    assert_array_equal(idx['hbo'], [4, 5])
    assert_array_equal(idx['hbr'], [6])


def _check_fwd_n_chan_consistent(fwd, n_expected):
    n_ok = len(fwd['info']['ch_names'])
    n_sol = fwd['sol']['data'].shape[0]
    assert_equal(n_expected, n_sol)
    assert_equal(n_expected, n_ok)


@testing.requires_testing_data
def test_pick_forward_seeg_ecog():
    """Test picking forward with SEEG and ECoG."""
    fwd = read_forward_solution(fname_meeg)
    counts = channel_indices_by_type(fwd['info'])
    for key in counts.keys():
        counts[key] = len(counts[key])
    counts['meg'] = counts['mag'] + counts['grad']
    fwd_ = pick_types_forward(fwd, meg=True)
    _check_fwd_n_chan_consistent(fwd_, counts['meg'])
    fwd_ = pick_types_forward(fwd, meg=False, eeg=True)
    _check_fwd_n_chan_consistent(fwd_, counts['eeg'])
    # should raise exception related to emptiness
    assert_raises(ValueError, pick_types_forward, fwd, meg=False, seeg=True)
    assert_raises(ValueError, pick_types_forward, fwd, meg=False, ecog=True)
    # change last chan from EEG to sEEG, second-to-last to ECoG
    ecog_name = 'E1'
    seeg_name = 'OTp1'
    rename_channels(fwd['info'], {'EEG 059': ecog_name})
    rename_channels(fwd['info'], {'EEG 060': seeg_name})
    for ch in fwd['info']['chs']:
        if ch['ch_name'] == seeg_name:
            ch['kind'] = FIFF.FIFFV_SEEG_CH
            ch['coil_type'] = FIFF.FIFFV_COIL_EEG
        elif ch['ch_name'] == ecog_name:
            ch['kind'] = FIFF.FIFFV_ECOG_CH
            ch['coil_type'] = FIFF.FIFFV_COIL_EEG
    fwd['sol']['row_names'][-1] = fwd['info']['chs'][-1]['ch_name']
    fwd['sol']['row_names'][-2] = fwd['info']['chs'][-2]['ch_name']
    counts['eeg'] -= 2
    counts['seeg'] += 1
    counts['ecog'] += 1
    # repick & check
    fwd_seeg = pick_types_forward(fwd, meg=False, seeg=True)
    assert_equal(fwd_seeg['sol']['row_names'], [seeg_name])
    assert_equal(fwd_seeg['info']['ch_names'], [seeg_name])
    # should work fine
    fwd_ = pick_types_forward(fwd, meg=True)
    _check_fwd_n_chan_consistent(fwd_, counts['meg'])
    fwd_ = pick_types_forward(fwd, meg=False, eeg=True)
    _check_fwd_n_chan_consistent(fwd_, counts['eeg'])
    fwd_ = pick_types_forward(fwd, meg=False, seeg=True)
    _check_fwd_n_chan_consistent(fwd_, counts['seeg'])
    fwd_ = pick_types_forward(fwd, meg=False, ecog=True)
    _check_fwd_n_chan_consistent(fwd_, counts['ecog'])


def test_picks_by_channels():
    """Test creating pick_lists."""
    rng = np.random.RandomState(909)

    test_data = rng.random_sample((4, 2000))
    ch_names = ['MEG %03d' % i for i in [1, 2, 3, 4]]
    ch_types = ['grad', 'mag', 'mag', 'eeg']
    sfreq = 250.0
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = RawArray(test_data, info)

    pick_list = _picks_by_type(raw.info)
    assert_equal(len(pick_list), 3)
    assert_equal(pick_list[0][0], 'mag')
    pick_list2 = _picks_by_type(raw.info, meg_combined=False)
    assert_equal(len(pick_list), len(pick_list2))
    assert_equal(pick_list2[0][0], 'mag')

    pick_list2 = _picks_by_type(raw.info, meg_combined=True)
    assert_equal(len(pick_list), len(pick_list2) + 1)
    assert_equal(pick_list2[0][0], 'meg')

    test_data = rng.random_sample((4, 2000))
    ch_names = ['MEG %03d' % i for i in [1, 2, 3, 4]]
    ch_types = ['mag', 'mag', 'mag', 'mag']
    sfreq = 250.0
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = RawArray(test_data, info)
    # This acts as a set, not an order
    assert_array_equal(pick_channels(info['ch_names'], ['MEG 002', 'MEG 001']),
                       [0, 1])

    # Make sure checks for list input work.
    assert_raises(ValueError, pick_channels, ch_names, 'MEG 001')
    assert_raises(ValueError, pick_channels, ch_names, ['MEG 001'], 'hi')

    pick_list = _picks_by_type(raw.info)
    assert_equal(len(pick_list), 1)
    assert_equal(pick_list[0][0], 'mag')
    pick_list2 = _picks_by_type(raw.info, meg_combined=True)
    assert_equal(len(pick_list), len(pick_list2))
    assert_equal(pick_list2[0][0], 'mag')

    # pick_types type check
    assert_raises(ValueError, raw.pick_types, eeg='string')


def test_clean_info_bads():
    """Test cleaning info['bads'] when bad_channels are excluded."""

    raw_file = op.join(op.dirname(__file__), 'io', 'tests', 'data',
                       'test_raw.fif')
    raw = read_raw_fif(raw_file)

    # select eeg channels
    picks_eeg = pick_types(raw.info, meg=False, eeg=True)

    # select 3 eeg channels as bads
    idx_eeg_bad_ch = picks_eeg[[1, 5, 14]]
    eeg_bad_ch = [raw.info['ch_names'][k] for k in idx_eeg_bad_ch]

    # select meg channels
    picks_meg = pick_types(raw.info, meg=True, eeg=False)

    # select randomly 3 meg channels as bads
    idx_meg_bad_ch = picks_meg[[0, 15, 34]]
    meg_bad_ch = [raw.info['ch_names'][k] for k in idx_meg_bad_ch]

    # simulate the bad channels
    raw.info['bads'] = eeg_bad_ch + meg_bad_ch

    # simulate the call to pick_info excluding the bad eeg channels
    info_eeg = pick_info(raw.info, picks_eeg)

    # simulate the call to pick_info excluding the bad meg channels
    info_meg = pick_info(raw.info, picks_meg)

    assert_equal(info_eeg['bads'], eeg_bad_ch)
    assert_equal(info_meg['bads'], meg_bad_ch)

    info = pick_info(raw.info, picks_meg)
    info._check_consistency()
    info['bads'] += ['EEG 053']
    assert_raises(RuntimeError, info._check_consistency)

run_tests_if_main()
