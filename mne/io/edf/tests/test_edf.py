"""Data Equivalence Tests"""
from __future__ import print_function

# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Martin Billinger <martin.billinger@tugraz.at>
#
# License: BSD (3-clause)

import os.path as op
import inspect
import warnings

from nose.tools import assert_equal, assert_true
from numpy.testing import assert_array_almost_equal, assert_array_equal
from numpy.testing import assert_raises, assert_allclose
from scipy import io
import numpy as np

from mne import pick_types, concatenate_raws
from mne.externals.six import iterbytes
from mne.utils import _TempDir
from mne.io import Raw, read_raw_edf
import mne.io.edf.edf as edfmodule
from mne.event import find_events

warnings.simplefilter('always')

FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
montage_path = op.join(data_dir, 'biosemi.hpts')
bdf_path = op.join(data_dir, 'test.bdf')
edf_path = op.join(data_dir, 'test.edf')
edf_uneven_path = op.join(data_dir, 'test_uneven_samp.edf')
bdf_eeglab_path = op.join(data_dir, 'test_bdf_eeglab.mat')
edf_eeglab_path = op.join(data_dir, 'test_edf_eeglab.mat')
edf_uneven_eeglab_path = op.join(data_dir, 'test_uneven_samp.mat')

eog = ['REOG', 'LEOG', 'IEOG']
misc = ['EXG1', 'EXG5', 'EXG8', 'M1', 'M2']


def test_bdf_data():
    """Test reading raw bdf files
    """
    raw_py = read_raw_edf(bdf_path, montage=montage_path, eog=eog,
                          misc=misc, preload=True)
    picks = pick_types(raw_py.info, meg=False, eeg=True, exclude='bads')
    data_py, _ = raw_py[picks]

    print(raw_py)  # to test repr
    print(raw_py.info)  # to test Info repr

    # this .mat was generated using the EEG Lab Biosemi Reader
    raw_eeglab = io.loadmat(bdf_eeglab_path)
    raw_eeglab = raw_eeglab['data'] * 1e-6  # data are stored in microvolts
    data_eeglab = raw_eeglab[picks]
    # bdf saved as a single, resolution to seven decimal points in matlab
    assert_array_almost_equal(data_py, data_eeglab, 8)

    # Manually checking that float coordinates are imported
    assert_true((raw_py.info['chs'][0]['eeg_loc']).any())
    assert_true((raw_py.info['chs'][25]['eeg_loc']).any())
    assert_true((raw_py.info['chs'][63]['eeg_loc']).any())

    # Make sure concatenation works
    raw_concat = concatenate_raws([raw_py.copy(), raw_py])
    assert_equal(raw_concat.n_times, 2 * raw_py.n_times)


def test_edf_data():
    """Test reading raw edf files
    """
    raw_py = read_raw_edf(edf_path, misc=range(-4, 0), stim_channel=139,
                          preload=True)

    picks = pick_types(raw_py.info, meg=False, eeg=True,
                       exclude=['EDF Annotations'])
    data_py, _ = raw_py[picks]

    print(raw_py)  # to test repr
    print(raw_py.info)  # to test Info repr

    # this .mat was generated using the EEG Lab Biosemi Reader
    raw_eeglab = io.loadmat(edf_eeglab_path)
    raw_eeglab = raw_eeglab['data'] * 1e-6  # data are stored in microvolts
    data_eeglab = raw_eeglab[picks]

    assert_array_almost_equal(data_py, data_eeglab, 10)

    # Make sure concatenation works
    raw_concat = concatenate_raws([raw_py.copy(), raw_py])
    assert_equal(raw_concat.n_times, 2 * raw_py.n_times)

    # Test uneven sampling
    raw_py = read_raw_edf(edf_uneven_path, stim_channel=None)
    data_py, _ = raw_py[0]
    # this .mat was generated using the EEG Lab Biosemi Reader
    raw_eeglab = io.loadmat(edf_uneven_eeglab_path)
    raw_eeglab = raw_eeglab['data']
    data_eeglab = raw_eeglab[0]

    # match upsampling
    upsample = len(data_eeglab) / len(raw_py)
    data_py = np.repeat(data_py, repeats=upsample)
    assert_array_equal(data_py, data_eeglab)


def test_read_segment():
    """Test writing raw edf files when preload is False
    """
    tempdir = _TempDir()
    raw1 = read_raw_edf(edf_path, stim_channel=None, preload=False)
    raw1_file = op.join(tempdir, 'test1-raw.fif')
    raw1.save(raw1_file, overwrite=True, buffer_size_sec=1)
    raw11 = Raw(raw1_file, preload=True)
    data1, times1 = raw1[:139, :]
    data11, times11 = raw11[:139, :]
    assert_allclose(data1, data11, rtol=1e-6)
    assert_array_almost_equal(times1, times11)
    assert_equal(sorted(raw1.info.keys()), sorted(raw11.info.keys()))

    raw2 = read_raw_edf(edf_path, stim_channel=None, preload=True)
    raw2_file = op.join(tempdir, 'test2-raw.fif')
    raw2.save(raw2_file, overwrite=True)
    data2, times2 = raw2[:139, :]
    assert_allclose(data1, data2, rtol=1e-6)
    assert_array_equal(times1, times2)

    raw1 = Raw(raw1_file, preload=True)
    raw2 = Raw(raw2_file, preload=True)
    assert_array_equal(raw1._data, raw2._data)

    # test the _read_segment function by only loading some of the data
    raw1 = read_raw_edf(edf_path, stim_channel=None, preload=False)
    raw2 = read_raw_edf(edf_path, stim_channel=None, preload=True)

    # select some random range of data to compare
    data1, times1 = raw1[:, 345:417]
    data2, times2 = raw2[:, 345:417]
    assert_array_equal(data1, data2)
    assert_array_equal(times1, times2)


def test_append():
    """Test appending raw edf objects using Raw.append
    """
    # Author: Alan Leggitt <alan.leggitt@ucsf.edu>
    raw = read_raw_edf(bdf_path, preload=False)
    raw0 = raw.copy()
    raw1 = raw.copy()
    raw0.append(raw1)
    assert_true(2 * len(raw) == len(raw0))


def test_parse_annotation():
    """Test parsing the tal channel
    """

    # test the parser
    annot = (b'+180\x14Lights off\x14Close door\x14\x00\x00\x00\x00\x00'
             b'+180\x14Lights off\x14\x00\x00\x00\x00\x00\x00\x00\x00'
             b'+180\x14Close door\x14\x00\x00\x00\x00\x00\x00\x00\x00'
             b'+3.14\x1504.20\x14nothing\x14\x00\x00\x00\x00'
             b'+1800.2\x1525.5\x14Apnea\x14\x00\x00\x00\x00\x00\x00\x00'
             b'+123\x14\x14\x00\x00\x00\x00\x00\x00\x00')
    annot = [a for a in iterbytes(annot)]
    annot[1::2] = [a * 256 for a in annot[1::2]]
    tal_channel = map(sum, zip(annot[0::2], annot[1::2]))
    events = edfmodule._parse_tal_channel(tal_channel)
    assert_equal(events, [[180.0, 0, 'Lights off'],
                          [180.0, 0, 'Close door'],
                          [180.0, 0, 'Lights off'],
                          [180.0, 0, 'Close door'],
                          [3.14, 4.2, 'nothing'],
                          [1800.2, 25.5, 'Apnea']])


def test_edf_annotations():
    """Test if events are detected correctly in a typical MNE workflow.
    """

    # test an actual file
    with warnings.catch_warnings(record=True):  # tal_channel dep
        warnings.simplefilter('always')
        raw = read_raw_edf(edf_path, tal_channel=-1, preload=True)
    edf_events = find_events(raw, output='step', shortest_event=0,
                             stim_channel='STI 014')

    # onset, duration, id
    events = [[0.1344, 0.2560, 2],
              [0.3904, 1.0000, 2],
              [2.0000, 0.0000, 3],
              [2.5000, 2.5000, 2]]
    events = np.array(events)
    events[:, :2] *= 512  # convert time to samples
    events = np.array(events, dtype=int)
    events[:, 1] -= 1
    events[events[:, 1] <= 0, 1] = 1
    events[:, 1] += events[:, 0]

    onsets = events[:, [0, 2]]
    offsets = events[:, [1, 2]]

    events = np.zeros((2 * events.shape[0], 3), dtype=int)
    events[0::2, [0, 2]] = onsets
    events[1::2, [0, 1]] = offsets

    assert_array_equal(edf_events, events)


def test_write_annotations():
    """Test writing raw files when annotations were parsed.
    """
    tempdir = _TempDir()
    with warnings.catch_warnings(record=True):  # tal_channel dep
        warnings.simplefilter('always')
        raw1 = read_raw_edf(edf_path, tal_channel=-1, preload=True)
    raw1_file = op.join(tempdir, 'test1-raw.fif')
    raw1.save(raw1_file, overwrite=True, buffer_size_sec=1)
    raw11 = Raw(raw1_file, preload=True)
    data1, times1 = raw1[:, :]
    data11, times11 = raw11[:, :]

    assert_array_almost_equal(data1, data11)
    assert_array_almost_equal(times1, times11)
    assert_equal(sorted(raw1.info.keys()), sorted(raw11.info.keys()))

    with warnings.catch_warnings(record=True):  # tal_channel dep
        warnings.simplefilter('always')
        assert_raises(RuntimeError, read_raw_edf,
                      edf_path, tal_channel=-1, preload=False)
