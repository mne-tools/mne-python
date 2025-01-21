# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
from ast import literal_eval
from datetime import datetime, timezone

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.io import loadmat

from mne.datasets.testing import data_path, requires_testing_data
from mne.io import read_raw_neuralynx
from mne.io.neuralynx.neuralynx import _exclude_kwarg
from mne.io.tests.test_raw import _test_raw_reader

testing_path = data_path(download=False) / "neuralynx"

pytest.importorskip("neo")


def _nlxheader_to_dict(matdict: dict) -> dict:
    """Convert the read-in "Header" field into a dict.

    All the key-value pairs of Header entries are formatted as strings
    (e.g. np.array("-AdbitVolts 0.000323513")) so we reformat that
    into dict by splitting at blank spaces.
    """
    entries = matdict["Header"][
        1::, :
    ]  # skip the first row which is just the "Header" string

    return {
        arr.item().item().split(" ")[0].strip("-"): arr.item().item().split(" ")[-1]
        for arr in entries
        if arr[0].size > 0
    }


def _read_nlx_mat_chan(matfile: str) -> np.ndarray:
    """Read a single channel from a Neuralynx .mat file."""
    mat = loadmat(matfile)

    hdr_dict = _nlxheader_to_dict(mat)

    # Nlx2MatCSC.m reads the data in N equal-sized (512-item) chunks
    # this array (1, n_chunks) stores the number of valid samples
    # per chunk (the last chunk is usually shorter)
    n_valid_samples = mat["NumberOfValidSamples"].ravel()

    # concatenate chunks, respecting the number of valid samples
    x = np.concatenate(
        [mat["Samples"][0:n, i] for i, n in enumerate(n_valid_samples)]
    )  # in ADBits

    # this value is the same for all channels and
    # converts data from ADBits to Volts
    conversionf = literal_eval(hdr_dict["ADBitVolts"])
    x = x * conversionf

    # if header says input was inverted at acquisition
    # (possibly for spike detection or so?), flip it back
    # NeuralynxIO does this under the hood in NeuralynxIO.parse_header()
    # see this discussion: https://github.com/NeuralEnsemble/python-neo/issues/819
    if hdr_dict["InputInverted"] == "True":
        x *= -1

    return x


def _read_nlx_mat_chan_keep_gaps(matfile: str) -> np.ndarray:
    """Read a single channel from a Neuralynx .mat file and keep invalid samples."""
    mat = loadmat(matfile)

    hdr_dict = _nlxheader_to_dict(mat)

    # Nlx2MatCSC.m reads the data in N equal-sized (512-item) chunks
    # this array (1, n_chunks) stores the number of valid samples
    # per chunk (the last chunk is usually shorter)
    n_valid_samples = mat["NumberOfValidSamples"].ravel()

    # read in the artificial zeros so that
    # we can compare with the mne padded arrays
    ncs_records_with_gaps = [9, 15, 20]
    for i in ncs_records_with_gaps:
        n_valid_samples[i] = 512

    # concatenate chunks, respecting the number of valid samples
    x = np.concatenate(
        [mat["Samples"][0:n, i] for i, n in enumerate(n_valid_samples)]
    )  # in ADBits

    # this value is the same for all channels and
    # converts data from ADBits to Volts
    conversionf = literal_eval(hdr_dict["ADBitVolts"])
    x = x * conversionf

    # if header says input was inverted at acquisition
    # (possibly for spike detection or so?), flip it back
    # NeuralynxIO does this under the hood in NeuralynxIO.parse_header()
    # see this discussion: https://github.com/NeuralEnsemble/python-neo/issues/819
    if hdr_dict["InputInverted"] == "True":
        x *= -1

    return x


# set known values for the Neuralynx data for testing
expected_chan_names = ["LAHC1", "LAHC2", "LAHC3", "xAIR1", "xEKG1"]
expected_hp_freq = 0.1
expected_lp_freq = 500.0
expected_sfreq = 2000.0
expected_meas_date = datetime.strptime("2023/11/02 13:39:27", "%Y/%m/%d  %H:%M:%S")


@requires_testing_data
def test_neuralynx():
    """Test basic reading."""
    from neo.io import NeuralynxIO

    excluded_ncs_files = [
        "LAHCu1.ncs",
        "LAHC1_3_gaps.ncs",
        "LAHC2_3_gaps.ncs",
    ]

    # ==== MNE-Python ==== #
    fname_patterns = ["*u*.ncs", "*3_gaps.ncs"]
    raw = read_raw_neuralynx(
        fname=testing_path,
        preload=True,
        exclude_fname_patterns=fname_patterns,
    )

    # test that we picked the right info from headers
    assert raw.info["highpass"] == expected_hp_freq, "highpass freq not set correctly"
    assert raw.info["lowpass"] == expected_lp_freq, "lowpass freq not set correctly"
    assert raw.info["sfreq"] == expected_sfreq, "sampling freq not set correctly"

    meas_date_utc = expected_meas_date.astimezone(timezone.utc)
    assert raw.info["meas_date"] == meas_date_utc, "meas_date not set correctly"

    # test that channel selection worked
    assert raw.ch_names == expected_chan_names, (
        "labels in raw.ch_names don't match expected channel names"
    )

    mne_y = raw.get_data()  # in V

    # ==== NeuralynxIO ==== #
    nlx_reader = NeuralynxIO(dirname=testing_path, **_exclude_kwarg(excluded_ncs_files))
    bl = nlx_reader.read(
        lazy=False
    )  # read a single block which contains the data split in segments

    # concatenate all signals and times from all segments (== total recording)
    nlx_y = np.concatenate(
        [sig.magnitude for seg in bl[0].segments for sig in seg.analogsignals]
    ).T
    nlx_y *= 1e-6  # convert from uV to V

    nlx_t = np.concatenate(
        [sig.times.magnitude for seg in bl[0].segments for sig in seg.analogsignals]
    ).T
    nlx_t = np.round(nlx_t, 3)  # round to millisecond precision

    nlx_ch_names = [ch[0] for ch in nlx_reader.header["signal_channels"]]

    # ===== Nlx2MatCSC.m ===== #
    matchans = ["LAHC1.mat", "LAHC2.mat", "LAHC3.mat", "xAIR1.mat", "xEKG1.mat"]

    # (n_chan, n_samples) array, in V
    mat_y = np.stack(
        [_read_nlx_mat_chan(os.path.join(testing_path, ch)) for ch in matchans]
    )

    # ===== Check sample values across MNE-Python, NeuralynxIO and MATLAB ===== #
    assert nlx_ch_names == raw.ch_names  # check channel names

    assert_allclose(
        mne_y, nlx_y, rtol=1e-6, err_msg="MNE and NeuralynxIO not all close"
    )  # data
    assert_allclose(
        mne_y, mat_y, rtol=1e-6, err_msg="MNE and Nlx2MatCSC.m not all close"
    )  # data

    _test_raw_reader(
        read_raw_neuralynx,
        fname=testing_path,
        exclude_fname_patterns=fname_patterns,
    )


@requires_testing_data
def test_neuralynx_gaps():
    """Test gap detection."""
    # ignore files with no gaps
    ignored_ncs_files = [
        "LAHC1.ncs",
        "LAHC2.ncs",
        "LAHC3.ncs",
        "xAIR1.ncs",
        "xEKG1.ncs",
        "LAHCu1.ncs",
    ]
    raw = read_raw_neuralynx(
        fname=testing_path,
        preload=True,
        exclude_fname_patterns=ignored_ncs_files,
    )
    mne_y, _ = raw.get_data(return_times=True)  # in V

    # there should be 2 channels with 3 gaps (of 130 samples in total)
    n_expected_gaps = 3
    n_expected_missing_samples = 130
    assert len(raw.annotations) == n_expected_gaps, "Wrong number of gaps detected"
    assert (mne_y[0, :] == 0).sum() == n_expected_missing_samples, (
        "Number of true and inferred missing samples differ"
    )

    # read in .mat files containing original gaps
    matchans = ["LAHC1_3_gaps.mat", "LAHC2_3_gaps.mat"]

    # (n_chan, n_samples) array, in V
    mat_y = np.stack(
        [
            _read_nlx_mat_chan_keep_gaps(os.path.join(testing_path, ch))
            for ch in matchans
        ]
    )

    # compare originally modified .ncs arrays with MNE-padded arrays
    # and test that we back-inserted 0's at the right places
    assert_allclose(
        mne_y, mat_y, rtol=1e-6, err_msg="MNE and Nlx2MatCSC.m not all close"
    )

    # test that channel selection works
    raw = read_raw_neuralynx(
        fname=testing_path,
        preload=False,
        exclude_fname_patterns=ignored_ncs_files,
    )

    raw.pick("LAHC2")
    assert raw.ch_names == ["LAHC2"]
    raw.load_data()  # before gh-12357 this would fail
