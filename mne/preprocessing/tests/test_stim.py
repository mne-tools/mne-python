# Authors: Daniel Strohmeier <daniel.strohmeier@tu-ilmenau.de>
#
# License: BSD-3-Clause

import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from mne.io import read_raw_fif
from mne.event import read_events
from mne.epochs import Epochs
from mne.preprocessing.stim import fix_stim_artifact

data_path = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_path, 'test_raw.fif')
event_fname = op.join(data_path, 'test-eve.fif')


def test_fix_stim_artifact():
    """Test fix stim artifact."""
    events = read_events(event_fname)

    raw = read_raw_fif(raw_fname)
    pytest.raises(RuntimeError, fix_stim_artifact, raw)

    raw = read_raw_fif(raw_fname, preload=True)

    # use window before stimulus in epochs
    tmin, tmax, event_id = -0.2, 0.5, 1
    picks = ('meg', 'eeg', 'eog')
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, reject=None)
    e_start = int(np.ceil(epochs.info['sfreq'] * epochs.tmin))
    tmin, tmax = -0.045, -0.015
    tmin_samp = int(-0.035 * epochs.info['sfreq']) - e_start
    tmax_samp = int(-0.015 * epochs.info['sfreq']) - e_start

    epochs = fix_stim_artifact(epochs, tmin=tmin, tmax=tmax, mode='linear',
                               picks=('eeg', 'eog'))
    data = epochs.copy().pick(
        ('eeg', 'eog')).get_data()[:, :, tmin_samp:tmax_samp]
    diff_data0 = np.diff(data[0][0])
    diff_data0 -= np.mean(diff_data0)
    assert_array_almost_equal(diff_data0, np.zeros(len(diff_data0)))

    data = epochs.copy().pick(('meg')).get_data()[:, :, tmin_samp:tmax_samp]
    diff_data0 = np.diff(data[0][0])
    diff_data0 -= np.mean(diff_data0)
    assert np.all(diff_data0 != 0)

    epochs = fix_stim_artifact(epochs, tmin=tmin, tmax=tmax, mode='window')
    data_from_epochs_fix = epochs.get_data()[:, :, tmin_samp:tmax_samp]
    assert not np.all(data_from_epochs_fix != 0)

    # use window before stimulus in raw
    event_idx = np.where(events[:, 2] == 1)[0][0]
    tmin, tmax = -0.045, -0.015
    tmin_samp = int(-0.035 * raw.info['sfreq'])
    tmax_samp = int(-0.015 * raw.info['sfreq'])
    tidx = int(events[event_idx, 0] - raw.first_samp)

    pytest.raises(ValueError, fix_stim_artifact, raw, events=np.array([]))
    raw = fix_stim_artifact(raw, events=None, event_id=1, tmin=tmin,
                            tmax=tmax, mode='linear', stim_channel='STI 014')
    data, times = raw[:, (tidx + tmin_samp):(tidx + tmax_samp)]
    diff_data0 = np.diff(data[0])
    diff_data0 -= np.mean(diff_data0)
    assert_array_almost_equal(diff_data0, np.zeros(len(diff_data0)))

    raw = fix_stim_artifact(raw, events, event_id=1, tmin=tmin,
                            tmax=tmax, mode='window')
    data, times = raw[:, (tidx + tmin_samp):(tidx + tmax_samp)]
    assert np.all(data) == 0.

    # get epochs from raw with fixed data
    tmin, tmax, event_id = -0.2, 0.5, 1
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, reject=None, baseline=None)
    e_start = int(np.ceil(epochs.info['sfreq'] * epochs.tmin))
    tmin_samp = int(-0.035 * epochs.info['sfreq']) - e_start
    tmax_samp = int(-0.015 * epochs.info['sfreq']) - e_start
    data_from_raw_fix = epochs.get_data()[:, :, tmin_samp:tmax_samp]
    assert np.all(data_from_raw_fix) == 0.

    # use window after stimulus
    evoked = epochs.average()
    tmin, tmax = 0.005, 0.045
    tmin_samp = int(0.015 * evoked.info['sfreq']) - evoked.first
    tmax_samp = int(0.035 * evoked.info['sfreq']) - evoked.first

    evoked = fix_stim_artifact(evoked, tmin=tmin, tmax=tmax, mode='linear')
    data = evoked.data[:, tmin_samp:tmax_samp]
    diff_data0 = np.diff(data[0])
    diff_data0 -= np.mean(diff_data0)
    assert_array_almost_equal(diff_data0, np.zeros(len(diff_data0)))

    evoked = fix_stim_artifact(evoked, tmin=tmin, tmax=tmax, mode='window')
    data = evoked.data[:, tmin_samp:tmax_samp]
    assert np.all(data) == 0.
