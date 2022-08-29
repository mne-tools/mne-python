# -*- coding: utf-8 -*-
# Authors: Adam Li  <adam2392@gmail.com>
#
# License: BSD-3-Clause

import os
import os.path as op
import shutil

import pytest
from numpy.testing import assert_array_equal
import numpy as np

from mne.datasets.testing import data_path, requires_testing_data
from mne.io import read_raw_persyst
from mne.io.tests.test_raw import _test_raw_reader

testing_path = data_path(download=False)
fname_lay = op.join(
    testing_path, 'Persyst',
    'sub-pt1_ses-02_task-monitor_acq-ecog_run-01_clip2.lay')
fname_dat = op.join(
    testing_path, 'Persyst',
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
def test_persyst_dates(tmp_path):
    """Test different Persyst date formats for meas date."""
    # now test what if you change contents of the lay file
    out_dir = str(tmp_path)
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
def test_persyst_wrong_file(tmp_path):
    """Test reading Persyst files when passed in wrong file path."""
    with pytest.raises(FileNotFoundError, match='The path you'):
        read_raw_persyst(fname_dat, preload=True)

    out_dir = str(tmp_path)
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
def test_persyst_moved_file(tmp_path):
    """Test reader - Persyst files need to be in same directory."""
    out_dir = str(tmp_path)
    new_fname_lay = op.join(out_dir, op.basename(fname_lay))
    new_fname_dat = op.join(out_dir, op.basename(fname_dat))
    shutil.copy(fname_lay, new_fname_lay)

    # original file read should work
    read_raw_persyst(fname_lay)

    # without a .dat file, reader should break
    # when the lay file was moved
    desired_err_msg = \
        'The data path you specified does ' \
        'not exist for the lay path, ' \
        'sub-pt1_ses-02_task-monitor_acq-ecog_run-01_clip2.lay'
    with pytest.raises(FileNotFoundError, match=desired_err_msg):
        read_raw_persyst(new_fname_lay, preload=True)

    # now change the file contents to point
    # to the full path, but it should still not work
    # as reader requires lay and dat file to be in
    # same directory
    with open(fname_lay, "r") as fin:
        with open(new_fname_lay, 'w') as fout:
            # for each line in the input file
            for idx, line in enumerate(fin):
                if line.startswith('File='):
                    # give it the full path to the old data
                    test_fpath = op.join(op.dirname(fname_dat),
                                         line.split('=')[1])
                    line = f'File={test_fpath}\n'
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
    new_fname_lay = tmp_path / op.basename(fname_lay)
    new_fname_dat = tmp_path / op.basename(fname_dat)
    shutil.copy(fname_dat, new_fname_dat)
    shutil.copy(fname_lay, new_fname_lay)

    raw = read_raw_persyst(new_fname_lay)
    raw.crop(tmin=0, tmax=4)

    # get the annotations and make sure that repeated annotations
    # are in the dataset
    annotations = raw.annotations
    assert np.count_nonzero(annotations.description == 'seizure') == 2

    # make sure annotation with a "," character is in there
    assert 'seizure1,2' in annotations.description
    assert 'CLip2' in annotations.description


@requires_testing_data
def test_persyst_errors(tmp_path):
    """Test reading Persyst files when passed in wrong file path."""
    out_dir = str(tmp_path)
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
