# Authors: Mark Wronkiewicz <wronk@uw.edu>
#          Qian Chu <qianchu99@gmail.com>
#
# License: BSD-3-Clause

import numpy as np
from numpy.testing import assert_allclose
from scipy.interpolate import interp1d
import pytest

from mne import create_info, find_events, Epochs, Annotations
from mne.io import RawArray
from mne.preprocessing import realign_raw


@pytest.mark.parametrize("ratio_other", (0.9, 0.99, 1, 1.01, 1.1))  # drifts
@pytest.mark.parametrize("start_raw, start_other", [(0, 0), (0, 3), (3, 0)])
@pytest.mark.parametrize("stop_raw, stop_other", [(0, 0), (0, 3), (3, 0)])
@pytest.mark.parametrize("test_annot", [True, False])
def test_realign(ratio_other, start_raw, start_other, stop_raw, stop_other, test_annot):
    """Test realigning raw."""
    # construct a true signal
    sfreq = 100.0
    duration = 50
    stop_raw = duration - stop_raw
    stop_other = duration - stop_other
    hann_len = 0.2
    box_len = 0.5
    signal = np.zeros(int(round((duration + 1) * sfreq)))
    orig_events = np.round(
        np.arange(max(start_raw, start_other) + 2, min(stop_raw, stop_other) - 2)
        * sfreq
    ).astype(int)
    signal[orig_events] = 1.0
    n_events = len(orig_events)
    times = np.arange(len(signal)) / sfreq
    stim = np.convolve(signal, np.ones(int(round(0.02 * sfreq))))[: len(times)]
    sig_hann = np.convolve(signal, np.hanning(int(round(hann_len * sfreq))))[
        : len(times)
    ]
    sig_box = np.convolve(signal, np.ones(int(round(box_len * sfreq))))[: len(times)]

    # construct our sampled versions of these signals (linear interp is fine)
    sfreq_raw = sfreq
    sfreq_other = ratio_other * sfreq
    raw_times = np.arange(start_raw, stop_raw, 1.0 / sfreq_raw)
    other_times = np.arange(start_other, stop_other, 1.0 / sfreq_other)
    assert raw_times[0] >= times[0]
    assert raw_times[-1] <= times[-1]
    assert other_times[0] >= times[0]
    assert other_times[-1] <= times[-1]
    data_raw = np.array(
        [
            interp1d(times, d, kind)(raw_times)
            for d, kind in (
                (stim, "nearest"),
                (sig_hann, "linear"),
                (sig_box, "linear"),
            )
        ]
    )
    data_other = np.array(
        [
            interp1d(times, d, kind)(other_times)
            for d, kind in (
                (stim, "nearest"),
                (sig_hann, "linear"),
                (sig_box, "linear"),
            )
        ]
    )
    info_raw = create_info(
        ["raw_stim", "raw_hann", "raw_box"], sfreq, ["stim", "eeg", "eeg"]
    )
    info_other = create_info(
        ["other_stim", "other_hann", "other_box"], sfreq, ["stim", "eeg", "eeg"]
    )
    raw = RawArray(data_raw, info_raw, first_samp=111)  # first_samp doesn't matter
    other = RawArray(data_other, info_other, first_samp=222)

    evoked_raw, events_raw, _, events_other = _assert_similarity(raw, other, n_events)

    if test_annot:
        onsets_raw = (events_raw[:, 0] - raw.first_samp) / raw.info["sfreq"]
        dur_raw = [box_len] * len(onsets_raw)
        desc_raw = ["raw_box"] * len(onsets_raw)
        annot_raw = Annotations(onsets_raw, dur_raw, desc_raw)
        raw.set_annotations(annot_raw)

        onsets_other = (events_other[:, 0] - other.first_samp) / other.info["sfreq"]
        dur_other = [box_len * ratio_other] * len(onsets_other)
        desc_other = ["other_box"] * len(onsets_other)
        annot_other = Annotations(onsets_other, dur_other, desc_other)
        other.set_annotations(annot_other)

    # realign
    t_raw = (events_raw[:, 0] - raw.first_samp) / raw.info["sfreq"]
    t_other = (events_other[:, 0] - other.first_samp) / other.info["sfreq"]
    assert duration - 10 <= len(events_raw) < duration
    raw_orig, other_orig = raw.copy(), other.copy()
    realign_raw(raw, other, t_raw, t_other, realign_annot=test_annot)

    # old events should still work for raw and produce the same result
    evoked_raw_2, events_raw, _, events_other = _assert_similarity(
        raw, other, n_events, events_raw=events_raw
    )
    assert_allclose(evoked_raw.data, evoked_raw_2.data)
    assert_allclose(raw.times, other.times)

    # raw data now aligned
    hann_corr = np.corrcoef(raw.get_data([1])[0], other.get_data([1])[0])[0, 1]
    assert 0.98 < hann_corr <= 1.0
    box_corr = np.corrcoef(raw.get_data([2])[0], other.get_data([2])[0])[0, 1]
    assert 0.98 < box_corr <= 1.0

    # annotations now aligned
    if test_annot:
        assert_allclose(
            raw.annotations.onset, events_raw[:, 0] / raw.info["sfreq"], atol=1 / sfreq
        )
        assert_allclose(
            other.annotations.onset,
            events_other[:, 0] / other.info["sfreq"],
            atol=1 / sfreq,
        )
        onsets_raw, dur_raw = (
            raw.annotations.onset - raw.first_samp / sfreq,
            raw.annotations.duration,
        )
        onsets_other, dur_other = (
            other.annotations.onset - other.first_samp / sfreq,
            raw.annotations.duration,
        )
        assert_allclose(onsets_raw, onsets_other, atol=1 / sfreq)
        assert_allclose(dur_raw, dur_other, atol=1 / sfreq)

    # Degenerate conditions -- only test in one run
    test_degenerate = (
        start_raw == start_other and stop_raw == stop_other and ratio_other == 1
    )
    if not test_degenerate:
        return
    # these alignments will not be correct but it shouldn't matter
    with pytest.warns(RuntimeWarning, match="^Fewer.*may be unreliable.*"):
        realign_raw(raw, other, raw_times[:5], other_times[:5])
    with pytest.raises(ValueError, match="same shape"):
        realign_raw(raw_orig, other_orig, raw_times[:5], other_times)
    rand_times = np.random.RandomState(0).randn(len(other_times))
    with pytest.raises(ValueError, match="cannot resample safely"):
        realign_raw(raw_orig, other_orig, rand_times, other_times)
    with pytest.warns(RuntimeWarning, match=".*computed as R=.*unreliable"):
        realign_raw(raw_orig, other_orig, raw_times + rand_times * 1000, other_times)


def _assert_similarity(raw, other, n_events, events_raw=None):
    if events_raw is None:
        events_raw = find_events(raw)
    events_other = find_events(other)
    assert len(events_raw) == n_events
    assert len(events_other) == n_events
    kwargs = dict(baseline=None, tmin=0, tmax=0.2)
    evoked_raw = Epochs(raw, events_raw, **kwargs).average()
    evoked_other = Epochs(other, events_other, **kwargs).average()
    assert evoked_raw.nave == evoked_other.nave == len(events_raw)
    assert len(evoked_raw.data) == len(evoked_other.data) == 2  # just EEG
    # corr = np.corrcoef(evoked_raw.data[0], evoked_other.data[0])[0, 1]
    # assert 0.9 <= corr <= 1.0 # may not hold when drift is large
    return evoked_raw, events_raw, evoked_other, events_other
