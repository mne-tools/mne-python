# -*- coding: utf-8 -*-
# Authors: Adam Li  <adam2392@gmail.com>
#          simplified BSD-3 license

import os.path as op
import shutil

import pytest

import mne
from mne.datasets.testing import data_path, requires_testing_data
from mne.io import read_raw_persyst
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


@requires_testing_data
def test_persyst_wrong_file():
    """Test reading Persyst files when passed in wrong file path."""
    with pytest.raises(FileNotFoundError, match='The path you'):
        read_raw_persyst(fname_dat, preload=True)

    out_dir = mne.utils._TempDir()
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


run_tests_if_main()
