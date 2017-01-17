"""Data Equivalence Tests"""
from __future__ import print_function

# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Martin Billinger <martin.billinger@tugraz.at>
#          Alan Leggitt <alan.leggitt@ucsf.edu>
#          Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import inspect
import warnings

from nose.tools import assert_equal, assert_true
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_raises)
from scipy import io
import numpy as np

from mne import pick_types
from mne.datasets import testing
from mne.externals.six import iterbytes
from mne.utils import run_tests_if_main, requires_pandas
from mne.io import read_raw_edf
from mne.io.tests.test_raw import _test_raw_reader
from mne.io.edf.edf import _parse_tal_channel
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
edf_stim_channel_path = op.join(data_dir, 'test_edf_stim_channel.edf')
edf_txt_stim_channel_path = op.join(data_dir, 'test_edf_stim_channel.txt')

data_path = testing.data_path(download=False)
edf_stim_resamp_path = op.join(data_path, 'EDF', 'test_edf_stim_resamp.edf')
edf_overlap_annot_path = op.join(data_path, 'EDF',
                                 'test_edf_overlapping_annotations.edf')
edf_reduced = op.join(data_path, 'EDF', 'test_reduced.edf')


eog = ['REOG', 'LEOG', 'IEOG']
misc = ['EXG1', 'EXG5', 'EXG8', 'M1', 'M2']


def test_bdf_data():
    """Test reading raw bdf files."""
    raw_py = _test_raw_reader(read_raw_edf, input_fname=bdf_path,
                              montage=montage_path, eog=eog, misc=misc,
                              exclude=['M2', 'IEOG'])
    assert_true('RawEDF' in repr(raw_py))
    picks = pick_types(raw_py.info, meg=False, eeg=True, exclude='bads')
    data_py, _ = raw_py[picks]

    # this .mat was generated using the EEG Lab Biosemi Reader
    raw_eeglab = io.loadmat(bdf_eeglab_path)
    raw_eeglab = raw_eeglab['data'] * 1e-6  # data are stored in microvolts
    data_eeglab = raw_eeglab[picks]
    # bdf saved as a single, resolution to seven decimal points in matlab
    assert_array_almost_equal(data_py, data_eeglab, 8)

    # Manually checking that float coordinates are imported
    assert_true((raw_py.info['chs'][0]['loc']).any())
    assert_true((raw_py.info['chs'][25]['loc']).any())
    assert_true((raw_py.info['chs'][63]['loc']).any())


@testing.requires_testing_data
def test_edf_overlapping_annotations():
    """Test EDF with overlapping annotations."""
    n_warning = 2
    with warnings.catch_warnings(record=True) as w:
        read_raw_edf(edf_overlap_annot_path, preload=True, verbose=True)
        assert_equal(sum('overlapping' in str(ww.message) for ww in w),
                     n_warning)


@testing.requires_testing_data
def test_edf_reduced():
    """Test EDF with various sampling rates."""
    _test_raw_reader(read_raw_edf, input_fname=edf_reduced, stim_channel=None)


def test_edf_data():
    """Test edf files."""
    raw = _test_raw_reader(read_raw_edf, input_fname=edf_path,
                           stim_channel=None, exclude=['Ergo-Left', 'H10'])
    raw_py = read_raw_edf(edf_path, preload=True)
    assert_equal(len(raw.ch_names) + 2, len(raw_py.ch_names))
    # Test saving and loading when annotations were parsed.
    edf_events = find_events(raw_py, output='step', shortest_event=0,
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


@testing.requires_testing_data
def test_stim_channel():
    """Test reading raw edf files with stim channel."""
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

    assert_raises(RuntimeError, read_raw_edf, edf_path, preload=False)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        raw = read_raw_edf(edf_stim_resamp_path, verbose=True)
    assert_equal(len(w), 1)
    assert_true('Events may jitter' in str(w[0].message))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        raw[:]
    assert_equal(len(w), 0)


def test_parse_annotation():
    """Test parsing the tal channel."""
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
    assert_equal(_parse_tal_channel([tal_channel]),
                 [[180.0, 0, 'Lights off'], [180.0, 0, 'Close door'],
                  [180.0, 0, 'Lights off'], [180.0, 0, 'Close door'],
                  [3.14, 4.2, 'nothing'], [1800.2, 25.5, 'Apnea']])


def test_edf_annotations():
    """Test if events are detected correctly in a typical MNE workflow."""
    # test an actual file
    raw = read_raw_edf(edf_path, preload=True)
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


def test_edf_stim_channel():
    """Test stim channel for edf file."""
    raw = read_raw_edf(edf_stim_channel_path, preload=True,
                       stim_channel=-1)
    true_data = np.loadtxt(edf_txt_stim_channel_path).T

    # EDF writer pad data if file to small
    _, ns = true_data.shape
    edf_data = raw._data[:, :ns]

    # assert stim channels are equal
    assert_array_equal(true_data[-1], edf_data[-1])

    # assert data are equal
    assert_array_almost_equal(true_data[0:-1] * 1e-6, edf_data[0:-1])


@requires_pandas
def test_to_data_frame():
    """Test edf Raw Pandas exporter."""
    for path in [edf_path, bdf_path]:
        raw = read_raw_edf(path, stim_channel=None, preload=True)
        _, times = raw[0, :10]
        df = raw.to_data_frame()
        assert_true((df.columns == raw.ch_names).all())
        assert_array_equal(np.round(times * 1e3), df.index.values[:10])
        df = raw.to_data_frame(index=None, scalings={'eeg': 1e13})
        assert_true('time' in df.index.names)
        assert_array_equal(df.values[:, 0], raw._data[0] * 1e13)


run_tests_if_main()
