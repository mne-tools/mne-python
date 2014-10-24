from nose.tools import assert_equal
from numpy.testing import assert_array_equal
from numpy import zeros, array
from mne import pick_channels_regexp, pick_types, Epochs
from mne.io.meas_info import create_info
from mne.io.array import RawArray
from mne.io.pick import (channel_indices_by_type, channel_type,
        pick_types_evoked, pick_types_forward)


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
