# -*- coding: utf-8 -*-
# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Martin Billinger <martin.billinger@tugraz.at>
#          Alan Leggitt <alan.leggitt@ucsf.edu>
#          Alexandre Barachant <alexandre.barachant@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)

import os.path as op
import inspect

import pytest

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal)
from scipy.io import loadmat

from mne import pick_types
from mne.datasets import testing
from mne.externals.six import iterbytes
from mne.utils import run_tests_if_main, requires_pandas, _TempDir
from mne.io import read_raw_edf
from mne.io.base import _RawShell
from mne.io.meas_info import _empty_info
from mne.io.tests.test_raw import _test_raw_reader
from mne.io.pick import channel_type
from mne.io.edf.edf import find_edf_events, _read_annot, _read_annotations_edf
from mne.io.edf.edf import _get_edf_default_event_id
from mne.io.edf.edf import _read_edf_header
from mne.event import find_events
from mne.annotations import events_from_annotations, read_annotations
from mne.io.meas_info import _kind_dict as _KIND_DICT

FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
montage_path = op.join(data_dir, 'biosemi.hpts')
bdf_path = op.join(data_dir, 'test.bdf')
edf_path = op.join(data_dir, 'test.edf')
duplicate_channel_labels_path = op.join(data_dir,
                                        'duplicate_channel_labels.edf')
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
bdf_stim_channel_path = op.join(data_path, 'BDF', 'test_bdf_stim_channel.bdf')


eog = ['REOG', 'LEOG', 'IEOG']
misc = ['EXG1', 'EXG5', 'EXG8', 'M1', 'M2']


def test_orig_units():
    """Test exposure of original channel units."""
    raw = read_raw_edf(edf_path, stim_channel='auto', preload=True)

    # Test original units
    orig_units = raw._orig_units
    assert len(orig_units) == len(raw.ch_names)
    assert orig_units['A1'] == u'µV'  # formerly 'uV' edit by _check_orig_units


def test_bdf_data():
    """Test reading raw bdf files."""
    raw_py = _test_raw_reader(read_raw_edf, input_fname=bdf_path,
                              eog=eog, misc=misc,
                              exclude=['M2', 'IEOG'], stim_channel=None)
    assert len(raw_py.ch_names) == 71
    raw_py = _test_raw_reader(read_raw_edf, input_fname=bdf_path,
                              montage=montage_path, eog=eog, misc=misc,
                              exclude=['M2', 'IEOG'], stim_channel=-1)
    assert len(raw_py.ch_names) == 71
    assert 'RawEDF' in repr(raw_py)
    picks = pick_types(raw_py.info, meg=False, eeg=True, exclude='bads')
    data_py, _ = raw_py[picks]

    # this .mat was generated using the EEG Lab Biosemi Reader
    raw_eeglab = loadmat(bdf_eeglab_path)
    raw_eeglab = raw_eeglab['data'] * 1e-6  # data are stored in microvolts
    data_eeglab = raw_eeglab[picks]
    # bdf saved as a single, resolution to seven decimal points in matlab
    assert_array_almost_equal(data_py, data_eeglab, 8)

    # Manually checking that float coordinates are imported
    assert (raw_py.info['chs'][0]['loc']).any()
    assert (raw_py.info['chs'][25]['loc']).any()
    assert (raw_py.info['chs'][63]['loc']).any()


@testing.requires_testing_data
def test_bdf_stim_channel():
    """Test BDF stim channel."""
    # test if last channel is detected as STIM by default
    raw_py = _test_raw_reader(read_raw_edf, input_fname=bdf_path,
                              stim_channel='auto')
    assert channel_type(raw_py.info, raw_py.info["nchan"] - 1) == 'stim'

    # test BDF file with wrong scaling info in header - this should be ignored
    # for BDF stim channels
    events = [[242, 0, 4],
              [310, 0, 2],
              [952, 0, 1],
              [1606, 0, 1],
              [2249, 0, 1],
              [2900, 0, 1],
              [3537, 0, 1],
              [4162, 0, 1],
              [4790, 0, 1]]
    raw = read_raw_edf(bdf_stim_channel_path, preload=True)
    bdf_events = find_events(raw)
    assert_array_equal(events, bdf_events)
    raw = read_raw_edf(bdf_stim_channel_path, preload=False,
                       stim_channel='auto')
    bdf_events = find_events(raw)
    assert_array_equal(events, bdf_events)


@testing.requires_testing_data
def test_edf_overlapping_annotations():
    """Test EDF with overlapping annotations."""
    with pytest.warns(RuntimeWarning, match='overlapping.* not fully support'):
        read_raw_edf(edf_overlap_annot_path, preload=True, stim_channel='auto',
                     verbose=True)


@testing.requires_testing_data
def test_edf_reduced():
    """Test EDF with various sampling rates."""
    _test_raw_reader(read_raw_edf, input_fname=edf_reduced, stim_channel=None,
                     verbose='error')


def test_edf_data():
    """Test edf files."""
    raw = _test_raw_reader(read_raw_edf, input_fname=edf_path,
                           stim_channel=None, exclude=['Ergo-Left', 'H10'],
                           verbose='error')
    raw_py = read_raw_edf(edf_path, stim_channel='auto', preload=True)

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

    # Test with number of records not in header (-1).
    tempdir = _TempDir()
    broken_fname = op.join(tempdir, 'broken.edf')
    with open(edf_path, 'rb') as fid_in:
        fid_in.seek(0, 2)
        n_bytes = fid_in.tell()
        fid_in.seek(0, 0)
        rbytes = fid_in.read(int(n_bytes * 0.4))
    with open(broken_fname, 'wb') as fid_out:
        fid_out.write(rbytes[:236])
        fid_out.write(bytes('-1      '.encode()))
        fid_out.write(rbytes[244:])
    with pytest.warns(RuntimeWarning,
                      match='records .* not match the file size'):
        raw = read_raw_edf(broken_fname, preload=True, stim_channel='auto')
        read_raw_edf(broken_fname, exclude=raw.ch_names[:132], preload=True,
                     stim_channel='auto')


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
    raw_eeglab = loadmat(edf_eeglab_path)
    raw_eeglab = raw_eeglab['data'] * 1e-6  # data are stored in microvolts
    data_eeglab = raw_eeglab[picks]

    assert_array_almost_equal(data_py, data_eeglab, 10)
    events = find_edf_events(raw_py)
    assert len(events) - 1 == len(find_events(raw_py))  # start not found

    # Test uneven sampling
    raw_py = read_raw_edf(edf_uneven_path, stim_channel=None)
    data_py, _ = raw_py[0]
    # this .mat was generated using the EEG Lab Biosemi Reader
    raw_eeglab = loadmat(edf_uneven_eeglab_path)
    raw_eeglab = raw_eeglab['data']
    data_eeglab = raw_eeglab[0]

    # match upsampling
    upsample = len(data_eeglab) / len(raw_py)
    data_py = np.repeat(data_py, repeats=upsample)
    assert_array_equal(data_py, data_eeglab)

    pytest.raises(RuntimeError, read_raw_edf, edf_path, preload=False,
                  stim_channel=-1)

    with pytest.warns(RuntimeWarning,
                      match='Interpolating stim .* Events may jitter'):
        raw = read_raw_edf(edf_stim_resamp_path, verbose=True, stim_channel=-1)
    with pytest.warns(None) as w:
        raw[:]
    assert len(w) == 0

    events = raw_py.find_edf_events()
    assert len(events) == 0


def test_duplicate_channel_labels_edf():
    """Test reading edf file with duplicate channel names."""
    EXPECTED_CHANNEL_NAMES = ['EEG F1-Ref-0', 'EEG F2-Ref', 'EEG F1-Ref-1']
    with pytest.warns(RuntimeWarning, match='Channel names are not unique'):
        raw = read_raw_edf(duplicate_channel_labels_path, preload=True)

    assert raw.ch_names[:-1] == EXPECTED_CHANNEL_NAMES


def test_parse_annotation(tmpdir):
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

    onset, duration, description = _read_annotations_edf([tal_channel])
    assert_equal(np.column_stack((onset, duration, description)),
                 [[180., 0., 'Lights off'], [180., 0., 'Close door'],
                  [180., 0., 'Lights off'], [180., 0., 'Close door'],
                  [3.14, 4.2, 'nothing'], [1800.2, 25.5, 'Apnea']])


def test_edf_annotations():
    """Test if events are detected correctly in a typical MNE workflow."""
    # test an actual file
    raw = read_raw_edf(edf_path, preload=True, stim_channel='auto')
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
    # test if stim channel is automatically detected
    raw = read_raw_edf(edf_path, preload=True, stim_channel='auto')
    assert channel_type(raw.info, raw.info["nchan"] - 1) == 'stim'

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
        raw = read_raw_edf(path, stim_channel=None, preload=True,
                           verbose='error')
        _, times = raw[0, :10]
        df = raw.to_data_frame()
        assert (df.columns == raw.ch_names).all()
        assert_array_equal(np.round(times * 1e3), df.index.values[:10])
        df = raw.to_data_frame(index=None, scalings={'eeg': 1e13})
        assert 'time' in df.index.names
        assert_array_equal(df.values[:, 0], raw._data[0] * 1e13)


def test_read_annot(tmpdir):
    """Test parsing the tal channel."""
    EXPECTED_ANNOTATIONS = [[180.0, 0, 'Lights off'], [180.0, 0, 'Close door'],
                            [180.0, 0, 'Lights off'], [180.0, 0, 'Close door'],
                            [3.14, 4.2, 'nothing'], [1800.2, 25.5, 'Apnea']]
    SFREQ = 100
    DATA_LENGTH = int(EXPECTED_ANNOTATIONS[-1][0] * SFREQ) + 1
    annot = (b'+180\x14Lights off\x14Close door\x14\x00\x00\x00\x00\x00'
             b'+180\x14Lights off\x14\x00\x00\x00\x00\x00\x00\x00\x00'
             b'+180\x14Close door\x14\x00\x00\x00\x00\x00\x00\x00\x00'
             b'+3.14\x1504.20\x14nothing\x14\x00\x00\x00\x00'
             b'+1800.2\x1525.5\x14Apnea\x14\x00\x00\x00\x00\x00\x00\x00'
             b'+123\x14\x14\x00\x00\x00\x00\x00\x00\x00')
    annot_file = tmpdir.join('annotations.txt')
    annot_file.write(annot)
    annotmap_file = tmpdir.join('annotations_map.txt')
    annotmap_file.write('Lights off:1,nothing:2,Apnea:3,Close door:4')

    stim_ch = _read_annot(annot=str(annot_file), annotmap=str(annotmap_file),
                          sfreq=SFREQ, data_length=DATA_LENGTH)

    assert stim_ch.shape == (DATA_LENGTH,)
    assert_array_equal(np.bincount(stim_ch), [180018, 0, 1, 1, 1])


def test_read_raw_edf_deprecation_of_annot_annotmap(tmpdir):
    """Test deprecation of annot and annotmap."""
    annot = (b'+0.1344\x150.2560\x14two\x14\x00\x00\x00\x00'
             b'+0.3904\x151.0\x14two\x14\x00\x00\x00\x00'
             b'+2.0\x14three\x14\x00\x00\x00\x00\x00\x00\x00\x00'
             b'+2.5\x152.5\x14two\x14\x00\x00\x00\x00')
    annot_file = tmpdir.join('annotations.txt')
    annot_file.write(annot)
    annotmap_file = tmpdir.join('annotations_map.txt')
    annotmap_file.write('two:2,three:3')

    with pytest.warns(DeprecationWarning, match="annot.*annotmap.*"):
        read_raw_edf(input_fname=edf_path, annot=str(annot_file),
                     annotmap=str(annotmap_file), preload=True)


def _compute_sfreq_from_edf_info(edf_info):
    # Compute sfreq from edf_info
    sel = edf_info['sel']
    n_samps = edf_info['n_samps'][sel]
    sfreq = n_samps.max() * \
        edf_info['record_length'][1] / edf_info['record_length'][0]

    return sfreq


def _get_empty_raw_with_valid_annot(fname):
    raw = _RawShell()
    raw.first_samp = 0
    edf_info, orig_units = _read_edf_header(fname=fname, annot=None,
                                            annotmap=None, exclude=())

    sfreq = _compute_sfreq_from_edf_info(edf_info)
    raw.info = _empty_info(sfreq)
    raw.info['meas_date'] = edf_info['meas_date']

    def _time_as_index(times, use_rounding, origin):
        if use_rounding:
            return np.round(np.atleast_1d(times) * sfreq)
        else:
            return np.floor(np.atleast_1d(times) * sfreq)

    raw.time_as_index = _time_as_index
    return raw


@testing.requires_testing_data
def test_find_events_and_events_from_annot_are_the_same():
    """Test that old behaviour and new produce the same events."""
    EXPECTED_EVENTS = [[68, 0, 2],
                       [199, 0, 2],
                       [1024, 0, 3],
                       [1280, 0, 2]]
    raw = read_raw_edf(edf_path, preload=True, stim_channel='auto')
    raw_shell = _get_empty_raw_with_valid_annot(edf_path)
    assert raw_shell.info['meas_date'] == raw.info['meas_date']
    assert raw_shell.info['sfreq'] == raw.info['sfreq']
    assert raw_shell.first_samp == raw.first_samp

    events_from_find_events = find_events(raw)
    assert_array_equal(events_from_find_events, EXPECTED_EVENTS)

    annot = read_annotations(edf_path)
    raw_shell.set_annotations(annot)
    event_id = _get_edf_default_event_id(annot.description)
    event_id.pop('start')
    events_from_EFA, _ = events_from_annotations(raw_shell, event_id=event_id,
                                                 use_rounding=False)

    assert_array_equal(events_from_EFA, events_from_find_events)


@pytest.mark.parametrize('EXPECTED, test_input', [
    pytest.param({'stAtUs': 'eeg', 'tRigGer': 'eeg', 'sine 1 Hz': 'eeg',
                  'STI 014': 'stim'}, 'auto', id='auto'),
    # pytest.param({'stAtUs': 'eeg', 'tRigGer': 'eeg', 'sine 1 Hz': 'eeg'},
    #              None, id='None'),
    # pytest.param({'stAtUs': 'eeg', 'tRigGer': 'eeg', 'sine 1 Hz': 'stim'},
    #              'sine 1 Hz', id='single string'),
    # pytest.param({'stAtUs': 'eeg', 'tRigGer': 'eeg', 'sine 1 Hz': 'stim'},
    #              2, id='single int'),
    pytest.param({'STI 014': 'stim', 'sine 1 Hz': 'eeg', 'stAtUs': 'eeg',
                  'tRigGer': 'eeg'}, -1, id='single int (revers indexing)')])
def test_edf_stim_ch_pick_up(test_input, EXPECTED):
    """Test stim_channel."""
    TYPE_LUT = {v[0]: k for k, v in _KIND_DICT.items()}
    fname = op.join(data_dir, 'test_stim_channel.edf')

    raw = read_raw_edf(fname, stim_channel=test_input, preload=True)
    ch_types = {ch['ch_name']: TYPE_LUT[ch['kind']] for ch in raw.info['chs']}
    assert ch_types == EXPECTED


run_tests_if_main()
