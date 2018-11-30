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
                           assert_equal)
from scipy.io import loadmat

import pytest

from mne import pick_types, Annotations
from mne.datasets import testing
from mne.utils import run_tests_if_main, requires_pandas, _TempDir
from mne.io import read_raw_edf
from mne.io.tests.test_raw import _test_raw_reader
from mne.io.edf.edf import _read_annotations_edf
from mne.io.edf.edf import _get_edf_default_event_id
from mne.annotations import events_from_annotations, read_annotations

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
bdf_stim_channel_path = op.join(data_path, 'BDF', 'test_bdf_stim_channel.bdf')

test_generator_bdf = op.join(data_dir, 'test_generator_2.bdf')
test_generator_edf = op.join(data_dir, 'test_generator_2.edf')

eog = ['REOG', 'LEOG', 'IEOG']
misc = ['EXG1', 'EXG5', 'EXG8', 'M1', 'M2']


def test_orig_units():
    """Test exposure of original channel units."""
    raw = read_raw_edf(edf_path, preload=True)

    # Test original units
    orig_units = raw._orig_units
    assert len(orig_units) == 140
    assert orig_units['A1'] == u'µV'  # formerly 'uV' edit by _check_orig_units


def test_bdf_data():
    """Test reading raw bdf files."""
    raw_py = _test_raw_reader(read_raw_edf, input_fname=bdf_path,
                              eog=eog, misc=misc,
                              exclude=['M2', 'IEOG'])
    assert len(raw_py.ch_names) == 71
    raw_py = _test_raw_reader(read_raw_edf, input_fname=bdf_path,
                              montage=montage_path, eog=eog, misc=misc,
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
def test_edf_reduced():
    """Test EDF with various sampling rates."""
    _test_raw_reader(read_raw_edf, input_fname=edf_reduced, verbose='error')


def test_edf_data():
    """Test edf files."""
    raw = _test_raw_reader(read_raw_edf, input_fname=edf_path,
                           exclude=['Ergo-Left', 'H10'], verbose='error')
    raw_py = read_raw_edf(edf_path, preload=True)

    assert_equal(len(raw.ch_names) + 2, len(raw_py.ch_names))

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
        fid_out.write(b'-1      ')
        fid_out.write(rbytes[244:])
    with pytest.warns(RuntimeWarning,
                      match='records .* not match the file size'):
        raw = read_raw_edf(broken_fname, preload=True)
        read_raw_edf(broken_fname, exclude=raw.ch_names[:132], preload=True)


def test_parse_annotation():
    """Test parsing the tal channel."""
    # test the parser
    annot = (b'+180\x14Lights off\x14Close door\x14\x00\x00\x00\x00\x00'
             b'+180\x14Lights off\x14\x00\x00\x00\x00\x00\x00\x00\x00'
             b'+180\x14Close door\x14\x00\x00\x00\x00\x00\x00\x00\x00'
             b'+3.14\x1504.20\x14nothing\x14\x00\x00\x00\x00'
             b'+1800.2\x1525.5\x14Apnea\x14\x00\x00\x00\x00\x00\x00\x00'
             b'+123\x14\x14\x00\x00\x00\x00\x00\x00\x00')
    annot = [a for a in bytes(annot)]
    annot[1::2] = [a * 256 for a in annot[1::2]]
    tal_channel = map(sum, zip(annot[0::2], annot[1::2]))

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
def test_to_data_frame():
    """Test edf Raw Pandas exporter."""
    for path in [edf_path, bdf_path]:
        raw = read_raw_edf(path, preload=True, verbose='error')
        _, times = raw[0, :10]
        df = raw.to_data_frame()
        assert (df.columns == raw.ch_names).all()
        assert_array_equal(np.round(times * 1e3), df.index.values[:10])
        df = raw.to_data_frame(index=None, scalings={'eeg': 1e13})
        assert 'time' in df.index.names
        assert_array_equal(df.values[:, 0], raw._data[0] * 1e13)


def test_read_raw_edf_deprecation():
    """Test edf raw reader deprecation."""
    _MSG = "`read_raw_edf` is not supposed to trigger a deprecation warning"
    with pytest.warns(None) as recwarn:
        read_raw_edf(edf_path)
    assert all([w.category != DeprecationWarning for w in recwarn.list]), _MSG

    with pytest.deprecated_call(match="stim_channel .* removed in 0.19"):
        read_raw_edf(edf_path, stim_channel=False)

    with pytest.raises(RuntimeError, match="stim channel is not supported"):
        read_raw_edf(edf_path, stim_channel='what ever')


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


def test_toy_bdf(recwarn):
    EXPECTED_EVENTS = ([6, 18, 24, 36, 200, 206, 224, 230] +
                       [x for x in range(400, 119801, 200)])
    annot = read_annotations(test_generator_bdf)
    assert len(annot.onset) == len(EXPECTED_EVENTS)+2


@pytest.mark.parametrize('fname', [test_generator_bdf, test_generator_edf])
def test_load_toy_examples_in_edf_branch(fname, recwarn):
    from mne.io.pick import channel_indices_by_type

    print(f'\n------------ fname: {fname} ---------')
    raw = read_raw_edf(test_generator_bdf)
    found_types = [k for k, v in
                   channel_indices_by_type(raw.info, picks=None).items()
                   if v]
    events, event_id = events_from_annotations(raw)
    print(f'The read types are {found_types}')
    print(f'The shape of the data is {raw.get_data().shape}')
    print(f"Channel names in _raw_extras: {raw._raw_extras[0]['ch_names']}")
    print(f'Annotations load in read_raw_edf: {raw.annotations}')
    print(f'Event ids found in Annotations: {event_id}')
    print(f'Events loaded:\n {events}')

    # raw.plot(scalings={k:'auto' for k in found_types},
    #          title=op.basename(fname))



run_tests_if_main()
