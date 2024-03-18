# Authors: Scott Huberty <seh33@uw.edu>
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest

import mne
from mne.datasets.testing import data_path, requires_testing_data
from mne.io import RawArray, read_raw_eyelink
from mne.preprocessing.eyetracking import deconvolve, interpolate_blinks

fname = data_path(download=False) / "eyetrack" / "test_eyelink.asc"
# pytest.importorskip("pandas")


@requires_testing_data
@pytest.mark.parametrize(
    "buffer, match, cause_error, interpolate_gaze",
    [
        (0.025, "BAD_blink", False, False),
        (0.025, "BAD_blink", False, True),
        ((0.025, 0.025), ["random_annot"], False, False),
        (0.025, "BAD_blink", True, False),
    ],
)
def test_interpolate_blinks(buffer, match, cause_error, interpolate_gaze):
    """Test interpolating pupil data during blinks."""
    raw = read_raw_eyelink(fname, create_annotations=["blinks"], find_overlaps=True)
    # Create a dummy stim channel
    # this will hit a certain line in the interpolate_blinks function
    info = mne.create_info(["STI"], raw.info["sfreq"], ["stim"])
    stim_data = np.zeros((1, len(raw.times)))
    stim_raw = RawArray(stim_data, info)
    raw.add_channels([stim_raw], force_update_info=True)

    # Get the indices of the first blink
    first_blink_start = raw.annotations[0]["onset"]
    first_blink_end = raw.annotations[0]["onset"] + raw.annotations[0]["duration"]
    if match == ["random_annot"]:
        msg = "No annotations matching"
        with pytest.warns(RuntimeWarning, match=msg):
            interpolate_blinks(raw, buffer=buffer, match=match)
        return

    if cause_error:
        # Make an annotation without ch_names info
        raw.annotations.append(onset=1, duration=1, description="BAD_blink")
        with pytest.raises(ValueError):
            interpolate_blinks(raw, buffer=buffer, match=match)
        return
    else:
        interpolate_blinks(
            raw, buffer=buffer, match=match, interpolate_gaze=interpolate_gaze
        )

    # Now get the data and check that the blinks are interpolated
    data, times = raw.get_data(return_times=True)
    # Get the indices of the first blink
    blink_ind = np.where((times >= first_blink_start) & (times <= first_blink_end))[0]
    # pupil data during blinks are zero, check that interpolated data are not zeros
    assert not np.any(data[2, blink_ind] == 0)  # left eye
    assert not np.any(data[5, blink_ind] == 0)  # right eye
    if interpolate_gaze:
        assert not np.isnan(data[0, blink_ind]).any()  # left eye
        assert not np.isnan(data[1, blink_ind]).any()  # right eye


@pytest.mark.parametrize("method", ["minimize", "inverse"])
def test_deconvolve(eyetrack_raw, method):
    """Test deconvolving pupil data."""
    # simulate the convolved pupil data
    signal = (np.zeros(30), np.ones(10), np.zeros(20), 2 * np.ones(10), np.zeros(30))
    signal = np.concatenate(signal, axis=0)
    kernel = np.exp(-(np.linspace(-2, 2, 20) ** 2))
    kernel = kernel / sum(kernel)
    ps = np.convolve(signal, kernel, mode="same")

    # replace the pupil data with the simulated data
    raw = eyetrack_raw.copy()
    raw.rename_channels({"pupil": "pupil_left"})
    raw._data[-1] = ps
    epochs = mne.make_fixed_length_epochs(raw, preload=True)

    data = epochs.get_data(picks=["pupil_left"])
    fit, times = deconvolve(epochs, method=method)
    # Check that we didn't change the epochs data in place
    np.testing.assert_array_equal(data, epochs.get_data(picks=["pupil_left"]))
    # Check that the fit has the expected shape
    n_chs = data.shape[1]
    np.testing.assert_equal(fit.shape, (len(epochs), n_chs, len(times)))
    # Check the above with custom spacing and bounds
    fit, times = deconvolve(epochs, spacing=[0, 0.3, 0.9], bounds=(0, np.inf))
    np.testing.assert_equal(fit.shape, (len(epochs), n_chs, len(times)))
    np.testing.assert_equal(len(times), 3)


def test_pupil_zscore(eyetrack_raw):
    """Test z-scoring pupil data."""
    signal = (np.zeros(30), np.ones(10), np.zeros(20), 2 * np.ones(10), np.zeros(30))
    signal = np.concatenate(signal, axis=0)
    raw = eyetrack_raw.copy()
    raw.rename_channels({"pupil": "pupil_left"})
    raw._data[-1] = signal
    epochs = mne.make_fixed_length_epochs(raw, preload=True)
    ps = mne.preprocessing.eyetracking.pupil_zscores(epochs, (None, None))
    np.testing.assert_almost_equal(ps.mean(), 0)
    np.testing.assert_almost_equal(ps.std(), 1)


def test_pupil_kernel():
    """Test the pupil kernel function."""
    kernel = mne.preprocessing.eyetracking.pupil_kernel(2, 2, 2, 2)
    np.testing.assert_equal(kernel.shape, (4,))
    kernel = mne.preprocessing.eyetracking.pupil_kernel(100, 1)
    t_kernel = np.linspace(0, 1, 100)
    np.testing.assert_almost_equal(t_kernel[np.argmax(kernel)], 0.93, decimal=2)
