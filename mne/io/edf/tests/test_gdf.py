# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#          Nicolas Barascud <nicolas.barascud@ens.fr>
#
# License: BSD (3-clause)

import os.path as op
import shutil

import pytest
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal)
import numpy as np
import scipy.io as sio

from mne.datasets import testing
from mne.io import read_raw_gdf
from mne.io.tests.test_raw import _test_raw_reader
from mne.utils import run_tests_if_main
from mne import pick_types, find_events, events_from_annotations

data_path = testing.data_path(download=False)
gdf1_path = op.join(data_path, 'GDF', 'test_gdf_1.25')
gdf2_path = op.join(data_path, 'GDF', 'test_gdf_2.20')
gdf_1ch_path = op.join(data_path, 'GDF', 'test_1ch.gdf')


@testing.requires_testing_data
def test_gdf_data():
    """Test reading raw GDF 1.x files."""
    raw = read_raw_gdf(gdf1_path + '.gdf', eog=None, misc=None, preload=True)
    picks = pick_types(raw.info, meg=False, eeg=True, exclude='bads')
    data, _ = raw[picks]

    # Test Status is added as event
    EXPECTED_EVS_ONSETS = raw._raw_extras[0]['events'][1]
    EXPECTED_EVS_ID = {
        '{}'.format(evs): i for i, evs in enumerate(
            [32769, 32770, 33024, 33025, 33026, 33027, 33028, 33029, 33040,
             33041, 33042, 33043, 33044, 33045, 33285, 33286], 1)
    }
    evs, evs_id = events_from_annotations(raw)
    assert_array_equal(evs[:, 0], EXPECTED_EVS_ONSETS)
    assert evs_id == EXPECTED_EVS_ID

    # this .npy was generated using the official biosig python package
    raw_biosig = np.load(gdf1_path + '_biosig.npy')
    raw_biosig = raw_biosig * 1e-6  # data are stored in microvolts
    data_biosig = raw_biosig[picks]

    # Assert data are almost equal
    assert_array_almost_equal(data, data_biosig, 8)

    # Test for events
    assert len(raw.annotations.duration == 963)

    # gh-5604
    assert raw.info['meas_date'] is None


@testing.requires_testing_data
def test_gdf2_birthday(tmpdir):
    """Test reading raw GDF 2.x files."""
    new_fname = str(tmpdir.join('temp.gdf'))
    shutil.copyfile(gdf2_path + '.gdf', new_fname)
    d = int(3.1e15)  # chosen by trial and error to give a reasonable age
    with open(new_fname, 'r+b') as fid:
        fid.seek(176, 0)
        assert np.fromfile(fid, np.uint64, 1)[0] == 0
        fid.seek(176, 0)
        fid.write(np.array([d], np.uint64).tobytes())
        fid.seek(176, 0)
        assert np.fromfile(fid, np.uint64, 1)[0] == d
    raw = read_raw_gdf(new_fname, eog=None, misc=None, preload=True)
    assert raw._raw_extras[0]['subject_info']['age'] == 44
    # XXX this is a bug, it should be populated...
    assert raw.info['subject_info'] is None


@testing.requires_testing_data
def test_gdf2_data():
    """Test reading raw GDF 2.x files."""
    raw = read_raw_gdf(gdf2_path + '.gdf', eog=None, misc=None, preload=True)
    assert raw._raw_extras[0]['subject_info']['age'] is None

    picks = pick_types(raw.info, meg=False, eeg=True, exclude='bads')
    data, _ = raw[picks]

    # This .mat was generated using the official biosig matlab package
    mat = sio.loadmat(gdf2_path + '_biosig.mat')
    data_biosig = mat['dat'] * 1e-6  # data are stored in microvolts
    data_biosig = data_biosig[picks]

    # Assert data are almost equal
    assert_array_almost_equal(data, data_biosig, 8)

    # Find events
    events = find_events(raw, verbose=1)
    events[:, 2] >>= 8  # last 8 bits are system events in biosemi files
    assert_equal(events.shape[0], 2)  # 2 events in file
    assert_array_equal(events[:, 2], [20, 28])

    # gh-5604
    assert raw.info['meas_date'] is None
    _test_raw_reader(read_raw_gdf, input_fname=gdf2_path + '.gdf',
                     eog=None, misc=None)


@testing.requires_testing_data
def test_one_channel_gdf():
    """Test a one-channel GDF file."""
    with pytest.warns(RuntimeWarning, match='different highpass'):
        ecg = read_raw_gdf(gdf_1ch_path, preload=True)
    assert ecg['ECG'][0].shape == (1, 4500)
    assert 150.0 == ecg.info['sfreq']


run_tests_if_main()
