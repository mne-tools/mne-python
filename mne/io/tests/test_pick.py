from nose.tools import assert_equal, assert_raises
from numpy.testing import assert_array_equal
import numpy as np
import os.path as op

from mne import (pick_channels_regexp, pick_types, Epochs,
                 read_forward_solution, rename_channels,
                 pick_info, pick_channels, __file__, create_info)
from mne.io import Raw, RawArray
from mne.io.pick import (channel_indices_by_type, channel_type,
                         pick_types_forward, _picks_by_type)
from mne.io.constants import FIFF
from mne.datasets import testing
from mne.utils import run_tests_if_main

data_path = testing.data_path(download=False)
fname_meeg = op.join(data_path, 'MEG', 'sample',
                     'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')


def test_pick_channels_regexp():
    """Test pick with regular expression
    """
    ch_names = ['MEG 2331', 'MEG 2332', 'MEG 2333']
    assert_array_equal(pick_channels_regexp(ch_names, 'MEG ...1'), [0])
    assert_array_equal(pick_channels_regexp(ch_names, 'MEG ...[2-3]'), [1, 2])
    assert_array_equal(pick_channels_regexp(ch_names, 'MEG *'), [0, 1, 2])


def test_pick_seeg():
    """Test picking with SEEG
    """
    names = 'A1 A2 Fz O OTp1 OTp2 OTp3'.split()
    types = 'mag mag eeg eeg seeg seeg seeg'.split()
    info = create_info(names, 1024., types)
    idx = channel_indices_by_type(info)
    assert_array_equal(idx['mag'], [0, 1])
    assert_array_equal(idx['eeg'], [2, 3])
    assert_array_equal(idx['seeg'], [4, 5, 6])
    assert_array_equal(pick_types(info, meg=False, seeg=True), [4, 5, 6])
    for i, t in enumerate(types):
        assert_equal(channel_type(info, i), types[i])
    raw = RawArray(np.zeros((len(names), 10)), info)
    events = np.array([[1, 0, 0], [2, 0, 0]])
    epochs = Epochs(raw, events, {'event': 0}, -1e-5, 1e-5)
    evoked = epochs.average(pick_types(epochs.info, meg=True, seeg=True))
    e_seeg = evoked.pick_types(meg=False, seeg=True, copy=True)
    for l, r in zip(e_seeg.ch_names, names[4:]):
        assert_equal(l, r)


def _check_fwd_n_chan_consistent(fwd, n_expected):
    n_ok = len(fwd['info']['ch_names'])
    n_sol = fwd['sol']['data'].shape[0]
    assert_equal(n_expected, n_sol)
    assert_equal(n_expected, n_ok)


@testing.requires_testing_data
def test_pick_forward_seeg():
    """Test picking forward with SEEG
    """
    fwd = read_forward_solution(fname_meeg)
    counts = channel_indices_by_type(fwd['info'])
    for key in counts.keys():
        counts[key] = len(counts[key])
    counts['meg'] = counts['mag'] + counts['grad']
    fwd_ = pick_types_forward(fwd, meg=True, eeg=False, seeg=False)
    _check_fwd_n_chan_consistent(fwd_, counts['meg'])
    fwd_ = pick_types_forward(fwd, meg=False, eeg=True, seeg=False)
    _check_fwd_n_chan_consistent(fwd_, counts['eeg'])
    # should raise exception related to emptiness
    assert_raises(ValueError, pick_types_forward, fwd, meg=False, eeg=False,
                  seeg=True)
    # change last chan from EEG to sEEG
    seeg_name = 'OTp1'
    rename_channels(fwd['info'], {'EEG 060': seeg_name})
    for ch in fwd['info']['chs']:
        if ch['ch_name'] == seeg_name:
            ch['kind'] = FIFF.FIFFV_SEEG_CH
            ch['coil_type'] = FIFF.FIFFV_COIL_EEG
    fwd['sol']['row_names'][-1] = fwd['info']['chs'][-1]['ch_name']
    counts['eeg'] -= 1
    counts['seeg'] += 1
    # repick & check
    fwd_seeg = pick_types_forward(fwd, meg=False, eeg=False, seeg=True)
    assert_equal(fwd_seeg['sol']['row_names'], [seeg_name])
    assert_equal(fwd_seeg['info']['ch_names'], [seeg_name])
    # should work fine
    fwd_ = pick_types_forward(fwd, meg=True, eeg=False, seeg=False)
    _check_fwd_n_chan_consistent(fwd_, counts['meg'])
    fwd_ = pick_types_forward(fwd, meg=False, eeg=True, seeg=False)
    _check_fwd_n_chan_consistent(fwd_, counts['eeg'])
    fwd_ = pick_types_forward(fwd, meg=False, eeg=False, seeg=True)
    _check_fwd_n_chan_consistent(fwd_, counts['seeg'])


def test_picks_by_channels():
    """Test creating pick_lists"""

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

    # Make sure checks for list input work.
    assert_raises(ValueError, pick_channels, ch_names, 'MEG 001')
    assert_raises(ValueError, pick_channels, ch_names, ['MEG 001'], 'hi')

    pick_list = _picks_by_type(raw.info)
    assert_equal(len(pick_list), 1)
    assert_equal(pick_list[0][0], 'mag')
    pick_list2 = _picks_by_type(raw.info, meg_combined=True)
    assert_equal(len(pick_list), len(pick_list2))
    assert_equal(pick_list2[0][0], 'mag')


def test_clean_info_bads():
    """Test cleaning info['bads'] when bad_channels are excluded """

    raw_file = op.join(op.dirname(__file__), 'io', 'tests', 'data',
                       'test_raw.fif')
    raw = Raw(raw_file)

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
    info = pick_info(raw.info, picks_meg)
    info._check_consistency()
    info['ch_names'][0] += 'f'
    assert_raises(RuntimeError, info._check_consistency)
    info = pick_info(raw.info, picks_meg)
    info._check_consistency()
    info['nchan'] += 1
    assert_raises(RuntimeError, info._check_consistency)

run_tests_if_main()
