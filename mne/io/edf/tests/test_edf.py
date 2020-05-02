# -*- coding: utf-8 -*-
# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Martin Billinger <martin.billinger@tugraz.at>
#          Alan Leggitt <alan.leggitt@ucsf.edu>
#          Alexandre Barachant <alexandre.barachant@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import inspect

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal, assert_allclose)
from scipy.io import loadmat

import pytest

from mne import pick_types, Annotations
from mne.datasets import testing
from mne.utils import run_tests_if_main, requires_pandas
from mne.io import read_raw_edf, read_raw_bdf
from mne.io.tests.test_raw import _test_raw_reader
from mne.io.edf.edf import _get_edf_default_event_id
from mne.io.edf.edf import _read_annotations_edf
from mne.io.edf.edf import _read_ch
from mne.io.edf.edf import _parse_prefilter_string
from mne.io.pick import channel_indices_by_type
from mne.annotations import events_from_annotations, read_annotations
from mne.io.meas_info import _kind_dict as _KIND_DICT


FILE = inspect.getfile(inspect.currentframe())
data_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
montage_path = op.join(data_dir, 'biosemi.hpts')  # XXX: missing reader
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
bdf_multiple_annotations_path = op.join(data_path, 'BDF',
                                        'multiple_annotation_chans.bdf')
test_generator_bdf = op.join(data_path, 'BDF', 'test_generator_2.bdf')
test_generator_edf = op.join(data_path, 'EDF', 'test_generator_2.edf')

eog = ['REOG', 'LEOG', 'IEOG']
misc = ['EXG1', 'EXG5', 'EXG8', 'M1', 'M2']


def test_orig_units():
    """Test exposure of original channel units."""
    raw = read_raw_edf(edf_path, preload=True)

    # Test original units
    orig_units = raw._orig_units
    assert len(orig_units) == len(raw.ch_names)
    assert orig_units['A1'] == 'µV'  # formerly 'uV' edit by _check_orig_units


def test_bdf_data():
    """Test reading raw bdf files."""
    raw_py = _test_raw_reader(read_raw_bdf, input_fname=bdf_path,
                              eog=eog, misc=misc,
                              exclude=['M2', 'IEOG'])
    assert len(raw_py.ch_names) == 71
    raw_py = _test_raw_reader(read_raw_bdf, input_fname=bdf_path,
                              montage='biosemi64', eog=eog, misc=misc,
                              exclude=['M2', 'IEOG'])
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
def test_bdf_crop_save_stim_channel(tmpdir):
    """Test EDF with various sampling rates."""
    raw = read_raw_bdf(bdf_stim_channel_path)
    raw.save(tmpdir.join('test-raw.fif'), tmin=1.2, tmax=4.0, overwrite=True)


@testing.requires_testing_data
@pytest.mark.parametrize('fname', [
    edf_reduced,
    edf_overlap_annot_path,
])
@pytest.mark.parametrize('stim_channel', (None, False, 'auto'))
def test_edf_others(fname, stim_channel):
    """Test EDF with various sampling rates and overlapping annotations."""
    _test_raw_reader(
        read_raw_edf, input_fname=fname, stim_channel=stim_channel,
        verbose='error')


def test_edf_data_broken(tmpdir):
    """Test edf files."""
    raw = _test_raw_reader(read_raw_edf, input_fname=edf_path,
                           exclude=['Ergo-Left', 'H10'], verbose='error')
    raw_py = read_raw_edf(edf_path)
    data = raw_py.get_data()
    assert_equal(len(raw.ch_names) + 2, len(raw_py.ch_names))

    # Test with number of records not in header (-1).
    broken_fname = op.join(tmpdir, 'broken.edf')
    with open(edf_path, 'rb') as fid_in:
        fid_in.seek(0, 2)
        n_bytes = fid_in.tell()
        fid_in.seek(0, 0)
        rbytes = fid_in.read()
    with open(broken_fname, 'wb') as fid_out:
        fid_out.write(rbytes[:236])
        fid_out.write(b'-1      ')
        fid_out.write(rbytes[244:244 + int(n_bytes * 0.4)])
    with pytest.warns(RuntimeWarning,
                      match='records .* not match the file size'):
        raw = read_raw_edf(broken_fname, preload=True)
        read_raw_edf(broken_fname, exclude=raw.ch_names[:132], preload=True)

    # Test with \x00's in the data
    with open(broken_fname, 'wb') as fid_out:
        fid_out.write(rbytes[:184])
        assert rbytes[184:192] == b'36096   '
        fid_out.write(rbytes[184:192].replace(b' ', b'\x00'))
        fid_out.write(rbytes[192:])
    raw_py = read_raw_edf(broken_fname)
    data_new = raw_py.get_data()
    assert_allclose(data, data_new)


def test_duplicate_channel_labels_edf():
    """Test reading edf file with duplicate channel names."""
    EXPECTED_CHANNEL_NAMES = ['EEG F1-Ref-0', 'EEG F2-Ref', 'EEG F1-Ref-1']
    with pytest.warns(RuntimeWarning, match='Channel names are not unique'):
        raw = read_raw_edf(duplicate_channel_labels_path, preload=False)

    assert raw.ch_names == EXPECTED_CHANNEL_NAMES


def test_parse_annotation(tmpdir):
    """Test parsing the tal channel."""
    # test the parser
    annot = (b'+180\x14Lights off\x14Close door\x14\x00\x00\x00\x00\x00'
             b'+180\x14Lights off\x14\x00\x00\x00\x00\x00\x00\x00\x00'
             b'+180\x14Close door\x14\x00\x00\x00\x00\x00\x00\x00\x00'
             b'+3.14\x1504.20\x14nothing\x14\x00\x00\x00\x00'
             b'+1800.2\x1525.5\x14Apnea\x14\x00\x00\x00\x00\x00\x00\x00'
             b'+123\x14\x14\x00\x00\x00\x00\x00\x00\x00')
    annot_file = tmpdir.join('annotations.txt')
    annot_file.write(annot)

    annot = [a for a in bytes(annot)]
    annot[1::2] = [a * 256 for a in annot[1::2]]
    tal_channel_A = np.array(list(map(sum, zip(annot[0::2], annot[1::2]))),
                             dtype=np.int64)

    with open(str(annot_file), 'rb') as fid:
        # ch_data = np.fromfile(fid, dtype=np.int16, count=len(annot))
        tal_channel_B = _read_ch(fid, subtype='EDF', dtype=np.int16,
                                 samp=(len(annot) - 1) // 2,
                                 dtype_byte='This_parameter_is_not_used')

    for tal_channel in [tal_channel_A, tal_channel_B]:
        onset, duration, description = _read_annotations_edf([tal_channel])
        assert_equal(np.column_stack((onset, duration, description)),
                     [[180., 0., 'Lights off'], [180., 0., 'Close door'],
                      [180., 0., 'Lights off'], [180., 0., 'Close door'],
                      [3.14, 4.2, 'nothing'], [1800.2, 25.5, 'Apnea']])


def test_find_events_backward_compatibility():
    """Test if events are detected correctly in a typical MNE workflow."""
    EXPECTED_EVENTS = [[68, 0, 2],
                       [199, 0, 2],
                       [1024, 0, 3],
                       [1280, 0, 2]]
    # test an actual file
    raw = read_raw_edf(edf_path, preload=True)
    event_id = _get_edf_default_event_id(raw.annotations.description)
    event_id.pop('start')
    events_from_EFA, _ = events_from_annotations(raw, event_id=event_id,
                                                 use_rounding=False)

    assert_array_equal(events_from_EFA, EXPECTED_EVENTS)


@requires_pandas
@pytest.mark.parametrize('fname', [edf_path, bdf_path])
def test_to_data_frame(fname):
    """Test EDF/BDF Raw Pandas exporter."""
    ext = op.splitext(fname)[1].lstrip('.').lower()
    if ext == 'edf':
        raw = read_raw_edf(fname, preload=True, verbose='error')
    elif ext == 'bdf':
        raw = read_raw_bdf(fname, preload=True, verbose='error')
    _, times = raw[0, :10]
    df = raw.to_data_frame(index='time')
    assert (df.columns == raw.ch_names).all()
    assert_array_equal(np.round(times * 1e3), df.index.values[:10])
    df = raw.to_data_frame(index=None, scalings={'eeg': 1e13})
    assert 'time' in df.columns
    assert_array_equal(df.values[:, 1], raw._data[0] * 1e13)


def test_read_raw_edf_stim_channel_input_parameters():
    """Test edf raw reader deprecation."""
    _MSG = "`read_raw_edf` is not supposed to trigger a deprecation warning"
    with pytest.warns(None) as recwarn:
        read_raw_edf(edf_path)
    assert all([w.category != DeprecationWarning for w in recwarn.list]), _MSG

    for invalid_stim_parameter in ['EDF Annotations', 'BDF Annotations']:
        with pytest.raises(ValueError,
                           match="stim channel is not supported"):
            read_raw_edf(edf_path, stim_channel=invalid_stim_parameter)


def _assert_annotations_equal(a, b):
    assert_array_equal(a.onset, b.onset)
    assert_array_equal(a.duration, b.duration)
    assert_array_equal(a.description, b.description)
    assert a.orig_time == b.orig_time


def test_read_annot(tmpdir):
    """Test parsing the tal channel."""
    EXPECTED_ANNOTATIONS = [[180.0, 0, 'Lights off'], [180.0, 0, 'Close door'],
                            [180.0, 0, 'Lights off'], [180.0, 0, 'Close door'],
                            [3.14, 4.2, 'nothing'], [1800.2, 25.5, 'Apnea']]

    EXPECTED_ONSET = [180.0, 180.0, 180.0, 180.0, 3.14, 1800.2]
    EXPECTED_DURATION = [0, 0, 0, 0, 4.2, 25.5]
    EXPECTED_DESC = ['Lights off', 'Close door', 'Lights off', 'Close door',
                     'nothing', 'Apnea']
    EXPECTED_ANNOTATIONS = Annotations(onset=EXPECTED_ONSET,
                                       duration=EXPECTED_DURATION,
                                       description=EXPECTED_DESC,
                                       orig_time=None)

    annot = (b'+180\x14Lights off\x14Close door\x14\x00\x00\x00\x00\x00'
             b'+180\x14Lights off\x14\x00\x00\x00\x00\x00\x00\x00\x00'
             b'+180\x14Close door\x14\x00\x00\x00\x00\x00\x00\x00\x00'
             b'+3.14\x1504.20\x14nothing\x14\x00\x00\x00\x00'
             b'+1800.2\x1525.5\x14Apnea\x14\x00\x00\x00\x00\x00\x00\x00'
             b'+123\x14\x14\x00\x00\x00\x00\x00\x00\x00')
    annot_file = tmpdir.join('annotations.txt')
    annot_file.write(annot)

    onset, duration, desc = _read_annotations_edf(annotations=str(annot_file))
    annotation = Annotations(onset=onset, duration=duration, description=desc,
                             orig_time=None)
    _assert_annotations_equal(annotation, EXPECTED_ANNOTATIONS)

    # Now test when reading from buffer of data
    with open(str(annot_file), 'rb') as fid:
        ch_data = np.fromfile(fid, dtype=np.int16, count=len(annot))
    onset, duration, desc = _read_annotations_edf([ch_data])
    annotation = Annotations(onset=onset, duration=duration, description=desc,
                             orig_time=None)
    _assert_annotations_equal(annotation, EXPECTED_ANNOTATIONS)


@testing.requires_testing_data
@pytest.mark.parametrize('fname', [test_generator_edf, test_generator_bdf])
def test_read_annotations(fname, recwarn):
    """Test IO of annotations from edf and bdf files via regexp."""
    annot = read_annotations(fname)
    assert len(annot.onset) == 2


def test_edf_prefilter_parse():
    """Test prefilter strings from header are parsed correctly."""
    prefilter_basic = ["HP: 0Hz LP: 0Hz"]
    highpass, lowpass = _parse_prefilter_string(prefilter_basic)
    assert_array_equal(highpass, ["0"])
    assert_array_equal(lowpass, ["0"])

    prefilter_normal_multi_ch = ["HP: 1Hz LP: 30Hz"] * 10
    highpass, lowpass = _parse_prefilter_string(prefilter_normal_multi_ch)
    assert_array_equal(highpass, ["1"] * 10)
    assert_array_equal(lowpass, ["30"] * 10)

    prefilter_unfiltered_ch = prefilter_normal_multi_ch + [""]
    highpass, lowpass = _parse_prefilter_string(prefilter_unfiltered_ch)
    assert_array_equal(highpass, ["1"] * 10)
    assert_array_equal(lowpass, ["30"] * 10)

    prefilter_edf_specs_doc = ["HP:0.1Hz LP:75Hz N:50Hz"]
    highpass, lowpass = _parse_prefilter_string(prefilter_edf_specs_doc)
    assert_array_equal(highpass, ["0.1"])
    assert_array_equal(lowpass, ["75"])


@testing.requires_testing_data
@pytest.mark.parametrize('fname', [test_generator_edf, test_generator_bdf])
def test_load_generator(fname, recwarn):
    """Test IO of annotations from edf and bdf files with raw info."""
    ext = op.splitext(fname)[1][1:].lower()
    if ext == 'edf':
        raw = read_raw_edf(fname)
    elif ext == 'bdf':
        raw = read_raw_bdf(fname)
    assert len(raw.annotations.onset) == 2
    found_types = [k for k, v in
                   channel_indices_by_type(raw.info, picks=None).items()
                   if v]
    assert len(found_types) == 1
    events, event_id = events_from_annotations(raw)
    ch_names = ['squarewave', 'ramp', 'pulse', 'ECG', 'noise', 'sine 1 Hz',
                'sine 8 Hz', 'sine 8.5 Hz', 'sine 15 Hz', 'sine 17 Hz',
                'sine 50 Hz']
    assert raw.get_data().shape == (11, 120000)
    assert raw.ch_names == ch_names
    assert event_id == {'RECORD START': 2, 'REC STOP': 1}
    assert_array_equal(events, [[0, 0, 2], [120000, 0, 1]])


@pytest.mark.parametrize('EXPECTED, test_input', [
    pytest.param({'stAtUs': 'stim', 'tRigGer': 'stim', 'sine 1 Hz': 'eeg'},
                 'auto', id='auto'),
    pytest.param({'stAtUs': 'eeg', 'tRigGer': 'eeg', 'sine 1 Hz': 'eeg'},
                 None, id='None'),
    pytest.param({'stAtUs': 'eeg', 'tRigGer': 'eeg', 'sine 1 Hz': 'stim'},
                 'sine 1 Hz', id='single string'),
    pytest.param({'stAtUs': 'eeg', 'tRigGer': 'eeg', 'sine 1 Hz': 'stim'},
                 2, id='single int'),
    pytest.param({'stAtUs': 'eeg', 'tRigGer': 'eeg', 'sine 1 Hz': 'stim'},
                 -1, id='single int (revers indexing)'),
    pytest.param({'stAtUs': 'stim', 'tRigGer': 'stim', 'sine 1 Hz': 'eeg'},
                 [0, 1], id='int list')])
def test_edf_stim_ch_pick_up(test_input, EXPECTED):
    """Test stim_channel."""
    # This is fragile for EEG/EEG-CSD, so just omit csd
    TYPE_LUT = {v[0]: k for k, v in _KIND_DICT.items() if k != 'csd'}
    fname = op.join(data_dir, 'test_stim_channel.edf')

    raw = read_raw_edf(fname, stim_channel=test_input)
    ch_types = {ch['ch_name']: TYPE_LUT[ch['kind']] for ch in raw.info['chs']}
    assert ch_types == EXPECTED


@testing.requires_testing_data
def test_bdf_multiple_annotation_channels():
    """Test BDF with multiple annotation channels."""
    raw = read_raw_bdf(bdf_multiple_annotations_path)
    assert len(raw.annotations) == 10
    descriptions = np.array(['signal_start', 'EEG-check#1', 'TestStim#1',
                             'TestStim#2', 'TestStim#3', 'TestStim#4',
                             'TestStim#5', 'TestStim#6', 'TestStim#7',
                             'Ligths-Off#1'], dtype='<U12')
    assert_array_equal(descriptions, raw.annotations.description)


run_tests_if_main()


@testing.requires_testing_data
def test_edf_lowpass_zero():
    """Test if a lowpass filter of 0Hz is mapped to the Nyquist frequency."""
    with pytest.warns(RuntimeWarning, match='too long.*truncated'):
        raw = read_raw_edf(edf_stim_resamp_path)
    assert_allclose(raw.info["lowpass"], raw.info["sfreq"] / 2)
