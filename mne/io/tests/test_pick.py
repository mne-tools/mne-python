from nose.tools import assert_equal, assert_raises
from numpy.testing import assert_array_equal
from numpy import zeros, array
from mne import (pick_channels_regexp, pick_types, Epochs, 
        read_forward_solution, rename_channels)
from mne.io.meas_info import create_info
from mne.io.array import RawArray
from mne.io.pick import (channel_indices_by_type, channel_type,
        pick_types_evoked, pick_types_forward)
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
    e_seeg = pick_types_evoked(evoked, meg=False, seeg=True)
    for l, r in zip(e_seeg.ch_names, names[4:]):
        assert_equal(l, r)


def _check_fwd_n_chan_consistent(fwd, n_expected):
    n_bad = len(fwd['info']['bads'])
    n_ok = len(fwd['info']['ch_names']) - n_bad
    n_sol = fwd['sol']['data'].shape[0] - n_bad
    assert_equal(n_expected, n_sol)
    assert_equal(n_expected, n_ok)


@testing.requires_testing_data
def test_pick_forward_seeg():
    fwd = read_forward_solution(test_forward.fname_meeg)
    # XXX non hard coded values?
    counts = {
        'meg': 305, 
        'eeg': 59, 
        'seeg': 0
    }
    types = counts.keys()
    # make convenient type selection kwds
    picks = dict()
    for t in types:
        picks[t] = dict()
        for t_ in types:
            picks[t][t_] = t_ == t
    for type in ('meg', 'eeg'):
        fwd_ = pick_types_forward(fwd, **picks[type])
        _check_fwd_n_chan_consistent(fwd_, counts[type])
    # should raise exception related to emptiness
    assert_raises(ValueError, pick_types_forward, fwd, **picks['seeg'])
    # change last chan from EEG to sEEG
    seeg_name = 'OTp1'
    rename_channels(fwd['info'], {'EEG 060': (seeg_name, 'seeg')})
    fwd['sol']['row_names'][-1] = fwd['info']['chs'][-1]['ch_name']
    counts['eeg'] -= 1
    counts['seeg'] += 1
    # repick & check
    fwd_seeg = pick_types_forward(fwd, **picks['seeg'])
    assert_equal(fwd_seeg['sol']['row_names'], [seeg_name])
    assert_equal(fwd_seeg['info']['ch_names'], [seeg_name])
    # should work fine
    for type, count in counts.items():
        fwd_ = pick_types_forward(fwd, **picks[type])
        _check_fwd_n_chan_consistent(fwd_, counts[type])
