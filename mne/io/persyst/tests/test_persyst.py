# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os
import shutil

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mne.datasets.testing import data_path, requires_testing_data
from mne.io import read_raw_persyst
from mne.io.tests.test_raw import _test_raw_reader
from mne.utils import _chmod_rw_R

testing_path = data_path(download=False)
fname_lay = (
    testing_path / "Persyst" / "sub-pt1_ses-02_task-monitor_acq-ecog_run-01_clip2.lay"
)
fname_dat = (
    testing_path / "Persyst" / "sub-pt1_ses-02_task-monitor_acq-ecog_run-01_clip2.dat"
)
# [SampleTimes] start of the testing file (seconds since midnight); it has 847 samples
# at 200 Hz
T0 = 9468.36


def _write_sample_times(tmp_path, sample_times):
    """Copy the testing recording, replacing its [SampleTimes] entries."""
    shutil.copy(fname_dat, tmp_path / fname_dat.name)
    text = fname_lay.read_text()
    text = text[: text.index("[SampleTimes]")]
    if sample_times is not None:
        text += "[SampleTimes]\n"
        for idx, time in sample_times:
            # strings are written as-is to allow malformed values
            time = f"{time:.3f}" if isinstance(time, float) else time
            text += f"{idx}={time}\n"
    fname = tmp_path / fname_lay.name
    fname.write_text(text)
    return fname


def _check_gaps(raw, gaps):
    """Check BAD_ACQ_SKIP annotations, zero-filled gaps, and the data around them."""
    orig = read_raw_persyst(fname_lay).get_data()
    sfreq = raw.info["sfreq"]
    skips = raw.annotations[raw.annotations.description == "BAD_ACQ_SKIP"]
    assert_allclose(skips.onset, [onset for onset, _ in gaps])
    assert_allclose(skips.duration, [duration for _, duration in gaps])
    assert raw.n_times == orig.shape[1] + round(sum(d for _, d in gaps) * sfreq)
    assert_array_equal(raw.get_data(reject_by_annotation="omit"), orig)
    for onset, duration in gaps:
        start = round(onset * sfreq)
        stop = start + round(duration * sfreq)
        assert_array_equal(raw.get_data(start=start, stop=stop), 0)


@requires_testing_data
def test_persyst_lay_load():
    """Test reading Persyst files using path to header file."""
    raw = read_raw_persyst(fname_lay, preload=False)

    # Test data import
    assert raw.info["sfreq"] == 200
    assert raw.preload is False

    # load raw data
    raw.load_data()
    assert raw._data.shape == (83, 847)
    assert raw.preload is True

    # defaults channels to EEG
    raw = raw.pick("eeg")
    assert len(raw.ch_names) == 83

    # no "-Ref" in channel names
    assert all(["-ref" not in ch.lower() for ch in raw.ch_names])

    # test with preload True
    raw = read_raw_persyst(fname_lay, preload=True)


@requires_testing_data
def test_persyst_raw():
    """Test reading Persyst files using path to header file."""
    raw = read_raw_persyst(fname_lay, preload=False)

    # defaults channels to EEG
    raw = raw.pick("eeg")

    # get data
    data, times = raw.get_data(start=200, return_times=True)
    assert data.shape == (83, 647)

    # seconds should match up to what is in the file
    assert times.min() == 1.0
    assert times.max() == 4.23

    # get data
    data = raw.get_data(start=200, stop=400)
    assert data.shape == (83, 200)

    # data should have been set correctly
    assert not data.min() == 0 and not data.max() == 0

    first_ch_data = raw.get_data(picks=[0], start=200, stop=400)
    assert_array_equal(first_ch_data.squeeze(), data[0, :])


@requires_testing_data
def test_persyst_dates(tmp_path):
    """Test different Persyst date formats for meas date."""
    # now test what if you change contents of the lay file
    new_fname_lay = tmp_path / fname_lay.name
    new_fname_dat = tmp_path / fname_dat.name
    shutil.copy(fname_dat, new_fname_dat)

    # reformat the lay file to have testdate with
    # "/" character
    with open(fname_lay) as fin:
        with open(new_fname_lay, "w") as fout:
            # for each line in the input file
            for idx, line in enumerate(fin):
                if line.startswith("TestDate"):
                    line = "TestDate=01/23/2000\n"
                fout.write(line)
    # file should update correctly with datetime
    raw = read_raw_persyst(new_fname_lay)
    assert raw.info["meas_date"].month == 1
    assert raw.info["meas_date"].day == 23
    assert raw.info["meas_date"].year == 2000

    # reformat the lay file to have testdate with
    # "-" character
    os.remove(new_fname_lay)
    with open(fname_lay) as fin:
        with open(new_fname_lay, "w") as fout:
            # for each line in the input file
            for idx, line in enumerate(fin):
                if line.startswith("TestDate"):
                    line = "TestDate=24-01-2000\n"
                fout.write(line)
    # file should update correctly with datetime
    raw = read_raw_persyst(new_fname_lay)
    assert raw.info["meas_date"].month == 1
    assert raw.info["meas_date"].day == 24
    assert raw.info["meas_date"].year == 2000


@requires_testing_data
def test_persyst_wrong_file(tmp_path):
    """Test reading Persyst files when passed in wrong file path."""
    with pytest.raises(FileNotFoundError, match="The path you"):
        read_raw_persyst(fname_dat, preload=True)

    new_fname_lay = tmp_path / fname_lay.name
    new_fname_dat = tmp_path / fname_dat.name
    shutil.copy(fname_lay, new_fname_lay)

    # without a .dat file, reader should break
    desired_err_msg = (
        "The data path you specified does "
        "not exist for the lay path, "
        "sub-pt1_ses-02_task-monitor_acq-ecog_run-01_clip2.lay"
    )
    with pytest.raises(FileNotFoundError, match=desired_err_msg):
        read_raw_persyst(new_fname_lay, preload=True)

    # once you copy over the .dat file things should work
    shutil.copy(fname_dat, new_fname_dat)
    read_raw_persyst(new_fname_lay, preload=True)


@requires_testing_data
def test_persyst_moved_file(tmp_path):
    """Test reader - Persyst files need to be in same directory."""
    new_fname_lay = tmp_path / fname_lay.name
    new_fname_dat = tmp_path / fname_dat.name
    shutil.copy(fname_lay, new_fname_lay)
    _chmod_rw_R(tmp_path)

    # original file read should work
    read_raw_persyst(fname_lay)

    # without a .dat file, reader should break
    # when the lay file was moved
    desired_err_msg = (
        "The data path you specified does "
        "not exist for the lay path, "
        "sub-pt1_ses-02_task-monitor_acq-ecog_run-01_clip2.lay"
    )
    with pytest.raises(FileNotFoundError, match=desired_err_msg):
        read_raw_persyst(new_fname_lay, preload=True)

    # now change the file contents to point
    # to the full path, but it should still not work
    # as reader requires lay and dat file to be in
    # same directory
    with open(fname_lay) as fin:
        with open(new_fname_lay, "w") as fout:
            # for each line in the input file
            for idx, line in enumerate(fin):
                if line.startswith("File="):
                    # give it the full path to the old data
                    test_fpath = fname_dat.parent / line.split("=")[1]
                    line = f"File={test_fpath}\n"
                fout.write(line)
    with pytest.raises(FileNotFoundError, match=desired_err_msg):
        read_raw_persyst(new_fname_lay, preload=True)

    # once we copy the dat file to the same directory, reader
    # should work
    shutil.copy(fname_dat, new_fname_dat)
    read_raw_persyst(new_fname_lay, preload=True)


@requires_testing_data
def test_persyst_standard():
    """Test standard operations."""
    _test_raw_reader(read_raw_persyst, fname=fname_lay)


@requires_testing_data
def test_persyst_annotations(tmp_path):
    """Test annotations reading in Persyst."""
    new_fname_lay = tmp_path / fname_lay.name
    new_fname_dat = tmp_path / fname_dat.name
    shutil.copy(fname_dat, new_fname_dat)
    shutil.copy(fname_lay, new_fname_lay)

    raw = read_raw_persyst(new_fname_lay)
    raw.crop(tmin=0, tmax=4)

    # get the annotations and make sure that repeated annotations
    # are in the dataset
    annotations = raw.annotations
    assert np.count_nonzero(annotations.description == "seizure") == 2

    # make sure annotation with a "," character is in there
    assert "seizure1,2" in annotations.description
    assert "CLip2" in annotations.description


@requires_testing_data
def test_persyst_errors(tmp_path):
    """Test reading Persyst files when passed in wrong file path."""
    new_fname_lay = tmp_path / fname_lay.name
    new_fname_dat = tmp_path / fname_dat.name
    shutil.copy(fname_dat, new_fname_dat)

    # reformat the lay file
    with open(fname_lay) as fin:
        with open(new_fname_lay, "w") as fout:
            # for each line in the input file
            for idx, line in enumerate(fin):
                if idx == 1:
                    line = line.replace("=", ",")
                fout.write(line)
    # file should break
    with pytest.raises(RuntimeError, match="The line"):
        read_raw_persyst(new_fname_lay)

    # reformat the lay file
    os.remove(new_fname_lay)
    with open(fname_lay) as fin:
        with open(new_fname_lay, "w") as fout:
            # for each line in the input file
            for idx, line in enumerate(fin):
                if line.startswith("WaveformCount"):
                    line = "WaveformCount=1\n"
                fout.write(line)
    # file should break
    with pytest.raises(RuntimeError, match="Channels in lay file do not"):
        read_raw_persyst(new_fname_lay)

    # reformat the lay file to have testdate
    # improperly specified
    os.remove(new_fname_lay)
    with open(fname_lay) as fin:
        with open(new_fname_lay, "w") as fout:
            # for each line in the input file
            for idx, line in enumerate(fin):
                if line.startswith("TestDate"):
                    line = "TestDate=Jan 23rd 2000\n"
                fout.write(line)
    # file should not read in meas date
    with pytest.warns(RuntimeWarning, match="Cannot read in the measurement date"):
        raw = read_raw_persyst(new_fname_lay)
        assert raw.info["meas_date"] is None


@requires_testing_data
def test_persyst_sampletimes_gap(tmp_path):
    """Test that [SampleTimes] gaps are zero-filled and annotated (gh-12339)."""
    orig = read_raw_persyst(fname_lay).get_data()
    # data resume at file sample 400 (2 s), 5 s later than if contiguous
    fname = _write_sample_times(tmp_path, [(0, T0), (400, T0 + 7)])
    raw = read_raw_persyst(fname, preload=True)
    assert raw.n_times == 847 + 1000
    data = raw.get_data()
    assert_array_equal(data[:, :400], orig[:, :400])
    assert_array_equal(data[:, 400:1400], 0)
    assert_array_equal(data[:, 1400:], orig[:, 400:])
    annot = raw.annotations
    assert annot.orig_time == raw.info["meas_date"]
    _check_gaps(raw, [(2.0, 5.0)])
    # comments before the gap are unchanged, comments after it are shifted
    assert_allclose(annot.onset[annot.description == "CLip2"], [0.0])
    assert_allclose(annot.onset[annot.description == "Clip1"], [3.73 + 5])

    # preload=False matches, including reads and crops straddling the gap
    raw_lazy = read_raw_persyst(fname, preload=False)
    assert_array_equal(raw_lazy.get_data(), data)
    assert_array_equal(raw_lazy.get_data(start=350, stop=1450), data[:, 350:1450])
    assert_array_equal(
        raw_lazy.copy().crop(1.5, 7.5).get_data(), raw.copy().crop(1.5, 7.5).get_data()
    )
    _test_raw_reader(read_raw_persyst, fname=fname)


@requires_testing_data
@pytest.mark.parametrize(
    "sample_times, gaps",
    [
        (None, []),  # no [SampleTimes] section
        ([(0, T0)], []),  # single entry (the testing file itself)
        ([(0, T0), (400, T0 + 2.001)], []),  # 1 ms timestamp jitter, contiguous
        ([(0, T0), (400, T0 + 2.005)], [(2.0, 0.005)]),  # 1-sample gap
        ([(100, T0 + 0.5), (400, T0 + 7)], [(2.0, 5.0)]),  # first entry not at 0
        ([(0, T0), (200, T0 + 2), (600, T0 + 6.5)], [(1.0, 1.0), (4.0, 2.5)]),
        ([(0, 86399.0), (400, 86406.0)], [(2.0, 5.0)]),  # values run past midnight
        ([(0, 86399.0), (400, 1.0)], []),  # wraps at midnight, no gap
        ([(0, 86399.0), (400, 6.0)], [(2.0, 5.0)]),  # wraps, gap across midnight
    ],
)
def test_persyst_sampletimes_layouts(tmp_path, sample_times, gaps):
    """Test [SampleTimes] layouts that read without warnings."""
    raw = read_raw_persyst(_write_sample_times(tmp_path, sample_times))
    _check_gaps(raw, gaps)
    if not gaps:  # no behavior change compared to the original file
        orig = read_raw_persyst(fname_lay)
        assert_array_equal(raw.get_data(), orig.get_data())
        assert_array_equal(raw.annotations.onset, orig.annotations.onset)
        assert_array_equal(raw.annotations.description, orig.annotations.description)


@requires_testing_data
def test_persyst_sampletimes_multiple_wraps(tmp_path):
    """Test that each midnight wrap is handled on its own."""
    # not in test_persyst_sampletimes_layouts, as _check_gaps would read ~17M samples
    # 86399 -> 0 wraps without a gap; 50000 -> 1 wraps again after a long gap
    sample_times = [(0, 86399.0), (200, 0.0), (300, 50000.0), (400, 1.0)]
    raw = read_raw_persyst(_write_sample_times(tmp_path, sample_times))
    skips = raw.annotations[raw.annotations.description == "BAD_ACQ_SKIP"]
    assert_allclose(skips.onset, [1.5, 1.5 + 49999.5 + 0.5])
    assert_allclose(skips.duration, [49999.5, 86400 + 1 - 50000 - 0.5])


@requires_testing_data
@pytest.mark.parametrize(
    "sample_times, match, gaps",
    [
        # clock steps back 10 s: treated as contiguous, the later gap is kept
        ([(0, T0), (200, T0 - 9), (400, T0 - 3)], "negative", [(2.0, 5.0)]),
        ([(0, T0), (400, "abc")], "Could not parse", []),
        ([(0, T0), (400, T0 + 7), (847, T0 + 20)], "beyond the end", [(2.0, 5.0)]),
    ],
)
def test_persyst_sampletimes_malformed(tmp_path, sample_times, match, gaps):
    """Test that malformed [SampleTimes] warn and fall back to contiguous data."""
    with pytest.warns(RuntimeWarning, match=match):
        raw = read_raw_persyst(_write_sample_times(tmp_path, sample_times))
    _check_gaps(raw, gaps)
