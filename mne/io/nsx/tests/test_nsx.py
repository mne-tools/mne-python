# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne import make_fixed_length_epochs
from mne._fiff.constants import FIFF
from mne._fiff.meas_info import _empty_info
from mne.datasets.testing import data_path, requires_testing_data
from mne.io import read_raw_nsx
from mne.io.nsx.nsx import _decode_online_filters, _read_header
from mne.io.tests.test_raw import _test_raw_reader

testing_path = data_path(download=False)
nsx_21_fname = os.path.join(testing_path, "nsx", "test_NEURALSG_raw.ns3")
nsx_22_fname = os.path.join(testing_path, "nsx", "test_NEURALCD_raw.ns3")
nsx_31_fname = os.path.join(testing_path, "nsx", "test_BRSMPGRP_raw.ns3")
nsx_test_fname = os.path.join(testing_path, "nsx", "Test_anonymized.ns3")
edf_test_fname = os.path.join(testing_path, "EDF", "test_reduced.edf")


def test_decode_online_filters():
    """Tests for online low/high-pass filter decoding."""
    info = _empty_info(100.0)
    highpass = np.array([0.0, 0.1])
    lowpass = np.array([50, 50])
    with pytest.warns(RuntimeWarning, match="different highpass filters"):
        _decode_online_filters(info, highpass, lowpass)
    assert info["highpass"] == 0.1

    info = _empty_info(100.0)
    highpass = np.array([0.0, 0.0])
    lowpass = np.array([40, 50])
    with pytest.warns(RuntimeWarning, match="different lowpass filters"):
        _decode_online_filters(info, highpass, lowpass)
    assert info["lowpass"] == 40

    info = _empty_info(100.0)
    highpass = np.array(["NaN", "NaN"])
    lowpass = np.array(["NaN", "NaN"])
    _decode_online_filters(info, highpass, lowpass)
    assert info["highpass"] == 0.0
    assert info["lowpass"] == 50.0


@requires_testing_data
def test_filetype_checks():
    """Tests for various error meaasges for wrong filetypes."""
    with pytest.raises(ValueError, match="does not match"):
        _read_header(edf_test_fname)
    with pytest.raises(NotImplementedError, match="Only NSx files"):
        read_raw_nsx(edf_test_fname)


@requires_testing_data
def test_nsx_ver_31():
    """Primary tests for BRSMPGRP reader."""
    raw = read_raw_nsx(nsx_31_fname)
    assert getattr(raw, "_data", False) is False
    assert raw.info["sfreq"] == 2000

    # Check info object
    assert raw.info["meas_date"].day == 31
    assert raw.info["meas_date"].year == 2023
    assert raw.info["meas_date"].month == 1
    assert raw.info["chs"][0]["cal"] == 0.6103515625
    assert raw.info["chs"][0]["range"] == 0.001

    # Check raw_extras
    for r in raw._raw_extras:
        assert r["orig_format"] == raw.orig_format
        assert r["orig_nchan"] == 128
        assert len(r["timestamp"]) == len(r["nb_data_points"])
        assert len(r["timestamp"]) == len(r["offset_to_data_block"])

    # Check annotations
    assert raw.annotations[0]["onset"] * raw.info["sfreq"] == 101
    assert raw.annotations[0]["duration"] * raw.info["sfreq"] == 49

    # Ignore following RuntimeWarning in mne/io/base.py in _write_raw_data
    # "Acquisition skips detected but did not fit evenly into output"
    # "buffer_size, will be written as zeroes."
    with pytest.warns(RuntimeWarning, match="skips detected"):
        raw = _test_raw_reader(
            read_raw_nsx,
            input_fname=nsx_31_fname,
            eog=None,
            misc=None,
            test_scaling=False,
            test_rank=False,
        )
    raw_data, _ = raw[:]
    n_channels, n_times = raw_data.shape
    assert n_times == 300
    assert n_channels == 128

    # Check data
    # There are two contiguous data packets (samples 0--100 and
    # samples 150--300. Each data was generated as:
    #     ```data = np.ones((n_samples, ch_count))
    #        data[n_samples // 2] = np.arange(ch_count) + 10
    #        data[:, ch_count // 2] = np.arange(n_samples) + 100```
    orig_data = raw_data / (raw.info["chs"][0]["cal"] * raw.info["chs"][0]["range"])
    assert_allclose(sum(orig_data[:, 50] - 10 - np.arange(n_channels)), 76.0)

    assert_allclose(orig_data[n_channels // 2, :100] - 100, np.arange(100))
    assert_allclose(orig_data[n_channels // 2, 150:] - 100, np.arange(150))

    data, times = raw.get_data(start=10, stop=20, return_times=True)
    assert 128, 10 == data.shape

    data, times = raw.get_data(start=50, stop=110, return_times=True)
    assert 128, 50 == data.shape

    data, times = raw.get_data(start=130, stop=170, return_times=True)
    assert 128, 40 == data.shape

    data, times = raw.get_data(start=200, stop=350, return_times=True)
    assert 128, 100 == data.shape

    data, times = raw.get_data(start=0, stop=300, return_times=True)
    epochs = make_fixed_length_epochs(raw, duration=0.05, preload=False)
    assert len(epochs.events) == 3
    epochs = make_fixed_length_epochs(raw, duration=0.05, preload=True)
    assert len(epochs) == 2
    assert "BAD_ACQ_SKIP" in epochs.drop_log[1]


@requires_testing_data
def test_nsx_ver_22():
    """Primary tests for NEURALCD reader."""
    raw = read_raw_nsx(
        nsx_22_fname,
    )
    assert getattr(raw, "_data", False) is False
    assert raw.info["sfreq"] == 2000

    # Check info object
    assert raw.info["meas_date"].day == 31
    assert raw.info["meas_date"].year == 2023
    assert raw.info["meas_date"].month == 1
    assert raw.info["chs"][0]["cal"] == 0.6103515625
    assert raw.info["chs"][0]["range"] == 0.001

    # check raw_extras
    for r in raw._raw_extras:
        assert r["orig_format"] == raw.orig_format
        assert r["orig_nchan"] == 128
        assert len(r["timestamp"]) == len(r["nb_data_points"])
        assert len(r["timestamp"]) == len(r["offset_to_data_block"])

    # Check annotations
    assert len(raw.annotations) == 0

    raw = _test_raw_reader(
        read_raw_nsx,
        input_fname=nsx_22_fname,
        eog=None,
        misc=None,
        test_scaling=False,  # XXX this should be True
        test_rank=False,
    )
    raw_data, _ = raw[:]
    n_channels, n_times = raw_data.shape
    assert n_times == 100
    assert n_channels == 128

    # Check data
    # There is only one contiguous data packet, samples 0--100. Data
    # was generated as:
    #     ```data = np.ones((n_samples, ch_count))
    #        data[n_samples // 2] = np.arange(ch_count) + 10
    #        data[:, ch_count // 2] = np.arange(n_samples) + 100```
    orig_data = raw_data / (raw.info["chs"][0]["cal"] * raw.info["chs"][0]["range"])
    assert_allclose(sum(orig_data[:, 50] - 10 - np.arange(n_channels)), 76.0)

    assert_allclose(orig_data[n_channels // 2, :100] - 100, np.arange(100))

    data, _ = raw.get_data(start=10, stop=20, return_times=True)
    assert 128, 10 == data.shape

    data, _ = raw.get_data(start=0, stop=300, return_times=True)
    assert 128, 100 == data.shape

    epochs = make_fixed_length_epochs(raw, duration=0.05, preload=True, id=1)
    assert len(epochs) == 1
    assert epochs.event_id["1"] == 1
    with pytest.raises(ValueError, match="No events produced"):
        _ = make_fixed_length_epochs(raw, duration=0.5, preload=True)


@requires_testing_data
def test_stim_eog_misc_chs_in_nsx():
    """Test stim/misc/eog channel assignments."""
    raw = read_raw_nsx(nsx_22_fname, stim_channel="elec127", eog=["elec126"])
    assert raw.info["chs"][127]["kind"] == FIFF.FIFFV_STIM_CH
    assert raw.info["chs"][126]["kind"] == FIFF.FIFFV_EOG_CH
    raw = read_raw_nsx(nsx_22_fname, stim_channel=["elec127"], eog=["elec126"])
    assert raw.info["chs"][127]["kind"] == FIFF.FIFFV_STIM_CH
    assert raw.info["chs"][126]["kind"] == FIFF.FIFFV_EOG_CH
    raw = read_raw_nsx(nsx_22_fname, stim_channel=127, eog=["elec126"])
    assert raw.info["chs"][127]["kind"] == FIFF.FIFFV_STIM_CH
    assert raw.info["chs"][126]["kind"] == FIFF.FIFFV_EOG_CH
    raw = read_raw_nsx(nsx_22_fname, stim_channel=[127], eog=["elec126"])
    assert raw.info["chs"][127]["kind"] == FIFF.FIFFV_STIM_CH
    assert raw.info["chs"][126]["kind"] == FIFF.FIFFV_EOG_CH
    stims = [ch_info["kind"] == FIFF.FIFFV_STIM_CH for ch_info in raw.info["chs"]]
    assert np.any(stims)
    assert raw.info["chs"][126]["kind"] == FIFF.FIFFV_EOG_CH
    with pytest.raises(ValueError, match="Invalid stim_channel"):
        raw = read_raw_nsx(nsx_22_fname, stim_channel=["elec128", 129], eog=["elec126"])
    with pytest.raises(ValueError, match="Invalid stim_channel"):
        raw = read_raw_nsx(nsx_22_fname, stim_channel=("elec128",), eog=["elec126"])

    raw = read_raw_nsx(nsx_22_fname, stim_channel="elec127", misc=["elec126", "elec1"])
    assert raw.info["chs"][126]["kind"] == FIFF.FIFFV_MISC_CH
    assert raw.info["chs"][1]["kind"] == FIFF.FIFFV_MISC_CH


@requires_testing_data
def test_nsx_ver_21():
    """Primary tests for NEURALSG reader."""
    with pytest.raises(NotImplementedError, match="(= NEURALSG)*not supported"):
        read_raw_nsx(nsx_21_fname)


@requires_testing_data
def test_nsx():
    """Tests for NEURALCD reader using real anonymized data."""
    raw = read_raw_nsx(
        nsx_test_fname,
    )
    assert getattr(raw, "_data", False) is False
    assert raw.info["sfreq"] == 2000

    # Check info object
    assert raw.info["meas_date"].day == 13
    assert raw.info["meas_date"].year == 2000
    assert raw.info["meas_date"].month == 6
    assert raw.info["lowpass"] == 1000
    assert raw.info["highpass"] == 0.3
    assert raw.info["chs"][0]["cal"] == 0.25
    assert raw.info["chs"][0]["range"] == 1e-6

    # check raw_extras
    for r in raw._raw_extras:
        assert r["orig_format"] == raw.orig_format
        assert r["orig_nchan"] == 5
        assert len(r["timestamp"]) == len(r["nb_data_points"])
        assert len(r["timestamp"]) == len(r["offset_to_data_block"])

    # Check annotations
    assert len(raw.annotations) == 0

    raw = _test_raw_reader(
        read_raw_nsx,
        input_fname=nsx_test_fname,
        eog=None,
        misc=None,
    )
    raw_data, times = raw[:]
    n_channels, n_times = raw_data.shape
    assert n_times == 100
    assert times.shape[0] == n_times
    assert n_channels == 5
    # Check data
    assert_allclose(
        raw_data.mean(axis=-1),
        np.array([-52.6375, 88.57, 70.5825, -22.055, -166.5]) * 1e-6,  # uV
    )
    assert raw.first_time == 3.8

    epochs = make_fixed_length_epochs(raw, duration=0.05, preload=True, id=1)
    assert len(epochs) == 1
    assert epochs.event_id["1"] == 1
    with pytest.raises(ValueError, match="No events produced"):
        _ = make_fixed_length_epochs(raw, duration=0.5, preload=True)
