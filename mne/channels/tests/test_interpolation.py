import os.path as op
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_raises, assert_equal

from mne import io, pick_types, read_events, Epochs
from mne.channels.interpolation import _make_interpolation_matrix


base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
evoked_nf_name = op.join(base_dir, 'test-nf-ave.fif')

event_id, tmin, tmax = 1, -0.2, 0.5
event_id_2 = 2

raw = io.Raw(raw_fname, add_eeg_ref=False)
events = read_events(event_name)
picks = pick_types(raw.info, meg=False, eeg=True, exclude=[])

reject = dict(eeg=80e-6)
epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                preload=True)
picks2 = pick_types(raw.info, meg=True, eeg=True, exclude=[])
epochs2 = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                 preload=True)[:1]


def test_interplation():
    """Test interpolation"""
    epochs.info['bads'] = []
    goods_idx = np.ones(len(epochs.ch_names), dtype=bool)
    goods_idx[epochs.ch_names.index('EEG 012')] = False
    bads_idx = ~goods_idx

    evoked = epochs.average()
    ave_before = evoked.data[bads_idx]

    pos = epochs.get_channel_positions()
    pos_good = pos[goods_idx]
    pos_bad = pos[bads_idx]
    interpolation = _make_interpolation_matrix(pos_good, pos_bad)
    assert_equal(interpolation.shape, (1, len(epochs.ch_names) - 1))

    ave_after = np.dot(interpolation, evoked.data[goods_idx])

    assert_array_almost_equal(ave_before, ave_after, decimal=5)

    epochs.info['bads'] = []
    assert_raises(ValueError, epochs.interpolate_bads_eeg)

    epochs.info['bads'] = ['EEG 012']
    epochs.preload = False
    assert_raises(ValueError,  epochs.interpolate_bads_eeg)

    epochs.preload = True
    epochs2 = epochs.copy()

    epochs2.interpolate_bads_eeg()
    ave_after2 = epochs2.average().data[bads_idx]

    assert_array_almost_equal(ave_after, ave_after2, decimal=16)

    raw = io.RawArray(data=epochs._data[0], info=epochs.info)
    raw_before = raw._data[bads_idx]
    raw.interpolate_bads_eeg()
    raw_after = raw._data[bads_idx]
    assert_equal(np.all(raw_before == raw_after), False)

    evoked = epochs.average()
    evoked.interpolate_bads_eeg()
    assert_array_equal(ave_after, evoked.data[bads_idx])

    for inst in [raw, epochs]:
        assert hasattr(inst, 'preload')
        inst.preload = False
        inst.info['bads'] = [inst.ch_names[1]]
        assert_raises(ValueError, inst.interpolate_bads_eeg)
