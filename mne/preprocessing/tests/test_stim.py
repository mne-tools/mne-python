# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from mne.epochs import Epochs
from mne.event import read_events
from mne.io import read_raw_fif
from mne.preprocessing.stim import fix_stim_artifact

data_path = Path(__file__).parents[2] / "io" / "tests" / "data"
raw_fname = data_path / "test_raw.fif"
event_fname = data_path / "test-eve.fif"


def test_fix_stim_artifact():
    """Test fix stim artifact."""
    events = read_events(event_fname)

    raw = read_raw_fif(raw_fname)
    pytest.raises(RuntimeError, fix_stim_artifact, raw)

    raw = read_raw_fif(raw_fname, preload=True)

    # use window before stimulus in epochs
    tmin, tmax, event_id = -0.2, 0.5, 1
    picks = ("meg", "eeg", "eog")
    epochs = Epochs(
        raw, events, event_id, tmin, tmax, picks=picks, preload=True, reject=None
    )
    e_start = int(np.ceil(epochs.info["sfreq"] * epochs.tmin))
    tmin, tmax = -0.045, -0.015
    tmin_samp = int(-0.035 * epochs.info["sfreq"]) - e_start
    tmax_samp = int(-0.015 * epochs.info["sfreq"]) - e_start

    epochs = fix_stim_artifact(
        epochs, tmin=tmin, tmax=tmax, mode="linear", picks=("eeg", "eog")
    )
    data = epochs.get_data(("eeg", "eog"))[:, :, tmin_samp:tmax_samp]
    diff_data0 = np.diff(data[0][0])
    diff_data0 -= np.mean(diff_data0)
    assert_array_almost_equal(diff_data0, np.zeros(len(diff_data0)))

    data = epochs.get_data("meg")[:, :, tmin_samp:tmax_samp]
    diff_data0 = np.diff(data[0][0])
    diff_data0 -= np.mean(diff_data0)
    assert np.all(diff_data0 != 0)

    epochs = fix_stim_artifact(epochs, tmin=tmin, tmax=tmax, mode="window")
    data_from_epochs_fix = epochs.get_data(copy=False)[:, :, tmin_samp:tmax_samp]
    assert not np.all(data_from_epochs_fix != 0)

    baseline = (-0.1, -0.05)
    epochs = fix_stim_artifact(
        epochs, tmin=tmin, tmax=tmax, baseline=baseline, mode="constant"
    )
    b_start = int(np.ceil(epochs.info["sfreq"] * baseline[0]))
    b_end = int(np.ceil(epochs.info["sfreq"] * baseline[1]))
    base_t1 = b_start - e_start
    base_t2 = b_end - e_start
    baseline_mean = epochs.get_data()[:, :, base_t1:base_t2].mean(axis=2)[0][0]
    data = epochs.get_data()[:, :, tmin_samp:tmax_samp]
    assert data[0][0][0] == baseline_mean

    # use window before stimulus in raw
    event_idx = np.where(events[:, 2] == 1)[0][0]
    tmin, tmax = -0.045, -0.015
    tmin_samp = int(-0.035 * raw.info["sfreq"])
    tmax_samp = int(-0.015 * raw.info["sfreq"])
    tidx = int(events[event_idx, 0] - raw.first_samp)

    pytest.raises(ValueError, fix_stim_artifact, raw, events=np.array([]))
    raw = fix_stim_artifact(
        raw,
        events=None,
        event_id=1,
        tmin=tmin,
        tmax=tmax,
        mode="linear",
        stim_channel="STI 014",
    )
    data, times = raw[:, (tidx + tmin_samp) : (tidx + tmax_samp)]
    diff_data0 = np.diff(data[0])
    diff_data0 -= np.mean(diff_data0)
    assert_array_almost_equal(diff_data0, np.zeros(len(diff_data0)))

    raw = fix_stim_artifact(
        raw, events, event_id=1, tmin=tmin, tmax=tmax, mode="window"
    )
    data, times = raw[:, (tidx + tmin_samp) : (tidx + tmax_samp)]

    assert np.all(data) == 0.0

    raw = fix_stim_artifact(
        raw,
        events,
        event_id=1,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        mode="constant",
    )
    data, times = raw[:, (tidx + tmin_samp) : (tidx + tmax_samp)]
    baseline_mean, _ = raw[:, (tidx + b_start) : (tidx + b_end)]
    assert baseline_mean.mean(axis=1)[0] == data[0][0]

    # get epochs from raw with fixed data
    tmin, tmax, event_id = -0.2, 0.5, 1
    epochs = Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        picks=picks,
        preload=True,
        reject=None,
        baseline=None,
    )
    e_start = int(np.ceil(epochs.info["sfreq"] * epochs.tmin))
    tmin_samp = int(-0.035 * epochs.info["sfreq"]) - e_start
    tmax_samp = int(-0.015 * epochs.info["sfreq"]) - e_start
    data_from_raw_fix = epochs.get_data(copy=False)[:, :, tmin_samp:tmax_samp]
    assert np.all(data_from_raw_fix) == 0.0

    # use window after stimulus
    evoked = epochs.average()
    tmin, tmax = 0.005, 0.045
    tmin_samp = int(0.015 * evoked.info["sfreq"]) - evoked.first
    tmax_samp = int(0.035 * evoked.info["sfreq"]) - evoked.first

    evoked = fix_stim_artifact(evoked, tmin=tmin, tmax=tmax, mode="linear")
    data = evoked.data[:, tmin_samp:tmax_samp]
    diff_data0 = np.diff(data[0])
    diff_data0 -= np.mean(diff_data0)
    assert_array_almost_equal(diff_data0, np.zeros(len(diff_data0)))

    evoked = fix_stim_artifact(evoked, tmin=tmin, tmax=tmax, mode="window")
    data = evoked.data[:, tmin_samp:tmax_samp]
    assert np.all(data) == 0.0

    evoked = fix_stim_artifact(
        evoked, tmin=tmin, tmax=tmax, baseline=baseline, mode="constant"
    )
    base_t1 = int(baseline[0] * evoked.info["sfreq"]) - evoked.first
    base_t2 = int(baseline[1] * evoked.info["sfreq"]) - evoked.first
    data = evoked.data[:, tmin_samp:tmax_samp]
    baseline_mean = evoked.data[:, base_t1:base_t2].mean(axis=1)[0]
    assert data[0][0] == baseline_mean
