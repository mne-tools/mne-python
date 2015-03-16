import os.path as op
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_allclose,
                           assert_array_equal)
from nose.tools import assert_raises, assert_equal

from mne import io, pick_types, read_events, Epochs
from mne.channels.interpolation import (_make_interpolation_matrix,
                                        _interpolate_bads_eeg_epochs)


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
                preload=True, reject=reject)

# make sure it works if MEG channels are present:
picks2 = np.concatenate([[0, 1, 2], picks])
n_meg = 3
epochs2 = Epochs(raw, events[epochs.selection], event_id, tmin, tmax,
                 picks=picks2, preload=True)


def test_interplation():
    """Test interpolation"""
    epochs_orig = epochs.copy()

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

    assert_allclose(ave_before, ave_after, atol=2e-6)

    epochs.info['bads'] = []
    assert_raises(ValueError, epochs.interpolate_bads_eeg)

    epochs.info['bads'] = ['EEG 012']
    epochs.preload = False
    assert_raises(ValueError,  epochs.interpolate_bads_eeg)
    epochs.preload = True

    epochs2.info['bads'] = ['EEG 012', 'MEG 1711']

    epochs2.interpolate_bads_eeg()
    ave_after2 = epochs2.average().data[n_meg + np.where(bads_idx)[0]]

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

    # test interpolating different channels per epoch
    epochsi = epochs_orig.copy()
    epochs_12 = epochs_orig.copy()
    epochs_12.info['bads'] = ['EEG 012']
    epochs_12.interpolate_bads_eeg()
    epochs_12_17 = epochs_orig.copy()
    epochs_12_17.info['bads'] = ['EEG 012', 'EEG 017']
    epochs_12_17.interpolate_bads_eeg()
    _interpolate_bads_eeg_epochs(epochsi, [['EEG 012'], [],
                                           ['EEG 012', 'EEG 017'], ['EEG 012'],
                                           [], [], []])
    assert_array_equal(epochsi._data[0], epochs_12._data[0])
    assert_array_equal(epochsi._data[1], epochs_orig._data[1])
    assert_array_equal(epochsi._data[2], epochs_12_17._data[2])
    assert_array_equal(epochsi._data[3], epochs_12._data[4])
    assert_array_equal(epochsi._data[4:], epochs_orig._data[4:])
