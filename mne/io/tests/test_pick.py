from nose.tools import assert_equal, assert_raises
from numpy.testing import assert_array_equal
import numpy as np
from numpy import zeros, array
import warnings

from mne import (pick_channels_regexp, pick_types, Epochs,
                 read_forward_solution, rename_channels)
from mne.io.meas_info import create_info
from mne.io.array import RawArray
from mne.io.pick import (channel_indices_by_type, channel_type,
                         pick_types_forward, _picks_by_type)
from mne.datasets import testing
from mne.forward.tests import test_forward


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
    raw = RawArray(zeros((len(names), 10)), info)
    events = array([[1, 0, 0], [2, 0, 0]]).astype('d')
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
    fwd = read_forward_solution(test_forward.fname_meeg)
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
    with warnings.catch_warnings(record=True):
        rename_channels(fwd['info'], {'EEG 060': (seeg_name, 'seeg')})
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

    pick_list = _picks_by_type(raw.info)
    assert_equal(len(pick_list), 1)
    assert_equal(pick_list[0][0], 'mag')
    pick_list2 = _picks_by_type(raw.info, meg_combined=True)
    assert_equal(len(pick_list), len(pick_list2))
    assert_equal(pick_list2[0][0], 'mag')
