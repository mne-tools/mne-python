"""Data Equivalence Tests"""
from __future__ import print_function

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

import os.path as op
import inspect

from nose.tools import assert_equal
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

import mne
from mne.utils import _TempDir
from mne import pick_types
from mne.io.constants import FIFF
from mne.io import Raw
from mne.io import read_raw_brainvision

FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
vhdr_path = op.join(data_dir, 'test.vhdr')
elp_path = op.join(data_dir, 'test_elp.txt')
eeg_bin = op.join(data_dir, 'test_bin_raw.fif')
elp_names = ['nasion', 'lpa', 'rpa', None, None, None, None, None,
             'FP1', 'FP2', 'F7', 'GND', 'F8',
             'FC5', 'F3', 'Fz', 'F4', 'FC6',
             'FC1', 'FCz', 'FC2', 'CP5', 'C3',
             'Cz', 'C4', 'CP6', 'CP1', 'CPz',
             'CP2', 'P7', 'P3', 'Pz', 'P4',
             'P8', 'O1', 'POz', 'O2', 'A1',
             'ReRef', 'HL', 'HR', 'Vb']
eog = ('HL', 'HR', 'Vb')

tempdir = _TempDir()


def test_brainvision_data():
    """Test reading raw Brain Vision files
    """
    raw_py = read_raw_brainvision(vhdr_path, elp_path, elp_names, preload=True)
    picks = pick_types(raw_py.info, meg=False, eeg=True, exclude='bads')
    data_py, times_py = raw_py[picks]

    print(raw_py)  # to test repr
    print(raw_py.info)  # to test Info repr

    # compare with a file that was generated using MNE-C
    raw_bin = Raw(eeg_bin, preload=True)
    picks = pick_types(raw_py.info, meg=False, eeg=True, exclude='bads')
    data_bin, times_bin = raw_bin[picks]

    assert_array_almost_equal(data_py, data_bin)
    assert_array_almost_equal(times_py, times_bin)

    # Make sure EOG channels are marked correctly
    raw_py = read_raw_brainvision(vhdr_path, elp_path, elp_names, eog=eog,
                                  preload=True)
    for ch in raw_py.info['chs']:
        if ch['ch_name'] in eog:
            assert_equal(ch['kind'], FIFF.FIFFV_EOG_CH)
        elif ch['ch_name'] in elp_names:
            assert_equal(ch['kind'], FIFF.FIFFV_EEG_CH)
        elif ch['ch_name'] == 'STI 014':
            assert_equal(ch['kind'], FIFF.FIFFV_STIM_CH)
        else:
            raise RuntimeError("Unknown Channel: %s" % ch['ch_name'])


def test_events():
    """Test reading and modifying events"""
    raw = read_raw_brainvision(vhdr_path, preload=True)

    # check that events are read and stim channel is synthesized correcly
    events = raw.get_brainvision_events()
    assert_array_equal(events, [[ 487, 1, 253],
                                [ 497, 1, 255],
                                [1770, 1, 254],
                                [1780, 1, 255],
                                [3253, 1, 254],
                                [3263, 1, 255],
                                [4936, 1, 253],
                                [4946, 1, 255],
                                [6620, 1, 254],
                                [6630, 1, 255]])

    mne_events = mne.find_events(raw, stim_channel='STI 014')
    assert_array_equal(events[:, [0, 2]], mne_events[:, [0, 2]])

    # modify events and check that stim channel is updated
    index = events[:, 2] == 255
    events = events[index]
    raw.set_brainvision_events(events)
    mne_events = mne.find_events(raw, stim_channel='STI 014')
    assert_array_equal(events[:, [0, 2]], mne_events[:, [0, 2]])

    # remove events
    nchan = raw.info['nchan']
    ch_name = raw.info['chs'][-2]['ch_name']
    events = np.empty((0, 3))
    raw.set_brainvision_events(events)
    assert_equal(raw.info['nchan'], nchan - 1)
    assert_equal(len(raw._data), nchan - 1)
    assert_equal(raw.info['chs'][-1]['ch_name'], ch_name)
    fname = op.join(tempdir, 'evt_raw.fif')
    raw.save(fname)

    # add events back in
    events = [[10, 1, 2]]
    raw.set_brainvision_events(events)
    assert_equal(raw.info['nchan'], nchan)
    assert_equal(len(raw._data), nchan)
    assert_equal(raw.info['chs'][-1]['ch_name'], 'STI 014')


def test_read_segment():
    """Test writing raw eeg files when preload is False
    """
    raw1 = read_raw_brainvision(vhdr_path, preload=False)
    raw1_file = op.join(tempdir, 'test1-raw.fif')
    raw1.save(raw1_file, overwrite=True)
    raw11 = Raw(raw1_file, preload=True)
    data1, times1 = raw1[:, :]
    data11, times11 = raw11[:, :]
    assert_array_almost_equal(data1, data11, 8)
    assert_array_almost_equal(times1, times11)
    assert_equal(sorted(raw1.info.keys()), sorted(raw11.info.keys()))

    raw2 = read_raw_brainvision(vhdr_path, preload=True)
    raw2_file = op.join(tempdir, 'test2-raw.fif')
    raw2.save(raw2_file, overwrite=True)
    data2, times2 = raw2[:, :]
    assert_array_equal(data1, data2)
    assert_array_equal(times1, times2)

    raw1 = Raw(raw1_file, preload=True)
    raw2 = Raw(raw2_file, preload=True)
    assert_array_equal(raw1._data, raw2._data)

    # save with buffer size smaller than file
    raw3_file = op.join(tempdir, 'test3-raw.fif')
    raw3 = read_raw_brainvision(vhdr_path)
    raw3.save(raw3_file, buffer_size_sec=2)
    raw3 = Raw(raw3_file, preload=True)
    assert_array_equal(raw3._data, raw1._data)

    # add reference channel
    raw4_file = op.join(tempdir, 'test4-raw.fif')
    raw4 = read_raw_brainvision(vhdr_path, reference='A1')
    raw4.save(raw4_file, buffer_size_sec=2)
    raw4 = Raw(raw4_file, preload=True)
    ref_idx = raw4.ch_names.index('A1')
    assert_equal(len(raw4._data), len(raw1._data) + 1)
    ref_data, _ = raw4[ref_idx]
    assert_array_equal(ref_data, 0)
