# -*- coding: utf-8 -*-
# Authors: Adam Li  <adam2392@gmail.com>
#          simplified BSD-3 license

import os
import os.path as op
import shutil

import pytest
from numpy.testing import assert_array_equal

import mne
from mne.datasets.testing import data_path, requires_testing_data
from mne.io import read_raw_persyst
from mne.io.tests.test_raw import _test_raw_reader
from mne.utils import run_tests_if_main

fname_lay = op.join(
    data_path(download=False), 'Persyst',
    'sub-pt1_ses-02_task-monitor_acq-ecog_run-01_clip2.lay')
fname_dat = op.join(
    data_path(download=False), 'Persyst',
    'sub-pt1_ses-02_task-monitor_acq-ecog_run-01_clip2.dat')


@requires_testing_data
def test_persyst_lay_load():
    """Test reading Persyst files using path to header file."""
    raw = read_raw_persyst(fname_lay, preload=False)

    # Test data import
    assert raw.info['sfreq'] == 200
    assert raw.preload is False

    # load raw data
    raw.load_data()
    assert raw._data.shape == (83, 847)
    assert raw.preload is True

    # defaults channels to EEG
    raw = raw.pick_types(eeg=True)
    assert len(raw.ch_names) == 83

    # no "-Ref" in channel names
    assert all(['-ref' not in ch.lower()
                for ch in raw.ch_names])

    # test with preload True
    raw = read_raw_persyst(fname_lay, preload=True)


@requires_testing_data
def test_persyst_raw():
    """Test reading Persyst files using path to header file."""
    raw = read_raw_persyst(fname_lay, preload=False)

    # defaults channels to EEG
    raw = raw.pick_types(eeg=True)

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
def test_persyst_dates():
    """Test different Persyst date formats for meas date."""
    # now test what if you change contents of the lay file
    out_dir = mne.utils._TempDir()
    new_fname_lay = op.join(out_dir, op.basename(fname_lay))
    new_fname_dat = op.join(out_dir, op.basename(fname_dat))
    shutil.copy(fname_dat, new_fname_dat)

    # reformat the lay file to have testdate with
    # "/" character
    with open(fname_lay, "r") as fin:
        with open(new_fname_lay, 'w') as fout:
            # for each line in the input file
            for idx, line in enumerate(fin):
                if line.startswith('TestDate'):
                    line = 'TestDate=01/23/2000\n'
                fout.write(line)
    # file should update correctly with datetime
    raw = read_raw_persyst(new_fname_lay)
    assert raw.info['meas_date'].month == 1
    assert raw.info['meas_date'].day == 23
    assert raw.info['meas_date'].year == 2000

    # reformat the lay file to have testdate with
    # "-" character
    os.remove(new_fname_lay)
    with open(fname_lay, "r") as fin:
        with open(new_fname_lay, 'w') as fout:
            # for each line in the input file
            for idx, line in enumerate(fin):
                if line.startswith('TestDate'):
                    line = 'TestDate=24-01-2000\n'
                fout.write(line)
    # file should update correctly with datetime
    raw = read_raw_persyst(new_fname_lay)
    assert raw.info['meas_date'].month == 1
    assert raw.info['meas_date'].day == 24
    assert raw.info['meas_date'].year == 2000


@requires_testing_data
def test_persyst_wrong_file(tmpdir):
    """Test reading Persyst files when passed in wrong file path."""
    with pytest.raises(FileNotFoundError, match='The path you'):
        read_raw_persyst(fname_dat, preload=True)

    out_dir = mne.utils._TempDir()
    out_dir = str(tmpdir)
    new_fname_lay = op.join(out_dir, op.basename(fname_lay))
    new_fname_dat = op.join(out_dir, op.basename(fname_dat))
    shutil.copy(fname_lay, new_fname_lay)

    # without a .dat file, reader should break
    desired_err_msg = \
        'The data path you specified does ' \
        'not exist for the lay path, ' \
        'sub-pt1_ses-02_task-monitor_acq-ecog_run-01_clip2.lay'
    with pytest.raises(FileNotFoundError, match=desired_err_msg):
        read_raw_persyst(new_fname_lay, preload=True)

    # once you copy over the .dat file things should work
    shutil.copy(fname_dat, new_fname_dat)
    read_raw_persyst(new_fname_lay, preload=True)


@requires_testing_data
def test_persyst_standard():
    """Test standard operations."""
    _test_raw_reader(read_raw_persyst, fname=fname_lay)


@requires_testing_data
def test_persyst_errors():
    """Test reading Persyst files when passed in wrong file path."""
    out_dir = mne.utils._TempDir()
    new_fname_lay = op.join(out_dir, op.basename(fname_lay))
    new_fname_dat = op.join(out_dir, op.basename(fname_dat))
    shutil.copy(fname_dat, new_fname_dat)

    # reformat the lay file
    with open(fname_lay, "r") as fin:
        with open(new_fname_lay, 'w') as fout:
            # for each line in the input file
            for idx, line in enumerate(fin):
                if idx == 1:
                    line = line.replace('=', ',')
                fout.write(line)
    # file should break
    with pytest.raises(RuntimeError, match='The line'):
        read_raw_persyst(new_fname_lay)

    # reformat the lay file
    os.remove(new_fname_lay)
    with open(fname_lay, "r") as fin:
        with open(new_fname_lay, 'w') as fout:
            # for each line in the input file
            for idx, line in enumerate(fin):
                if line.startswith('WaveformCount'):
                    line = 'WaveformCount=1\n'
                fout.write(line)
    # file should break
    with pytest.raises(RuntimeError, match='Channels in lay '
                                           'file do not'):
        read_raw_persyst(new_fname_lay)

    # reformat the lay file
    os.remove(new_fname_lay)
    with open(fname_lay, "r") as fin:
        with open(new_fname_lay, 'w') as fout:
            # for each line in the input file
            for idx, line in enumerate(fin):
                if line.startswith('File'):
                    line = f'File=/{op.basename(fname_dat)}\n'
                fout.write(line)
    # file should break
    with pytest.raises(FileNotFoundError, match='The data path '
                                                'you specified'):
        read_raw_persyst(new_fname_lay)

    # reformat the lay file to have testdate
    # improperly specified
    os.remove(new_fname_lay)
    with open(fname_lay, "r") as fin:
        with open(new_fname_lay, 'w') as fout:
            # for each line in the input file
            for idx, line in enumerate(fin):
                if line.startswith('TestDate'):
                    line = 'TestDate=Jan 23rd 2000\n'
                fout.write(line)
    # file should not read in meas date
    with pytest.warns(RuntimeWarning,
                      match='Cannot read in the measurement date'):
        raw = read_raw_persyst(new_fname_lay)
        assert raw.info['meas_date'] is None


run_tests_if_main()
