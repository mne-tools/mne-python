# -*- coding: utf-8 -*-
# Authors: Denis A. Engemann  <denis.engemann@gmail.com>
#          simplified BSD-3 license


import os.path as op
import inspect

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
import pytest
from scipy import io as sio


from mne import find_events, pick_types
from mne.io import read_raw_egi
from mne.io.tests.test_raw import _test_raw_reader
from mne.io.egi.egi import _combine_triggers
from mne.utils import run_tests_if_main
from mne.datasets.testing import data_path, requires_testing_data

FILE = inspect.getfile(inspect.currentframe())
base_dir = op.join(op.dirname(op.abspath(FILE)), 'data')
egi_fname = op.join(base_dir, 'test_egi.raw')
egi_txt_fname = op.join(base_dir, 'test_egi.txt')


@requires_testing_data
def test_io_egi_mff():
    """Test importing EGI MFF simple binary files."""
    egi_fname_mff = op.join(data_path(), 'EGI', 'test_egi.mff')
    raw = read_raw_egi(egi_fname_mff, include=None)
    assert ('RawMff' in repr(raw))
    include = ['DIN1', 'DIN2', 'DIN3', 'DIN4', 'DIN5', 'DIN7']
    raw = _test_raw_reader(read_raw_egi, input_fname=egi_fname_mff,
                           include=include, channel_naming='EEG %03d')

    assert_equal('eeg' in raw, True)
    eeg_chan = [c for c in raw.ch_names if 'EEG' in c]
    assert_equal(len(eeg_chan), 129)
    picks = pick_types(raw.info, eeg=True)
    assert_equal(len(picks), 129)
    assert_equal('STI 014' in raw.ch_names, True)

    events = find_events(raw, stim_channel='STI 014')
    assert_equal(len(events), 8)
    assert_equal(np.unique(events[:, 1])[0], 0)
    assert (np.unique(events[:, 0])[0] != 0)
    assert (np.unique(events[:, 2])[0] != 0)

    pytest.raises(ValueError, read_raw_egi, egi_fname_mff, include=['Foo'],
                  preload=False)
    pytest.raises(ValueError, read_raw_egi, egi_fname_mff, exclude=['Bar'],
                  preload=False)
    for ii, k in enumerate(include, 1):
        assert (k in raw.event_id)
        assert (raw.event_id[k] == ii)


def test_io_egi():
    """Test importing EGI simple binary files."""
    # test default
    with open(egi_txt_fname) as fid:
        data = np.loadtxt(fid)
    t = data[0]
    data = data[1:]
    data *= 1e-6  # µV

    with pytest.warns(RuntimeWarning, match='Did not find any event code'):
        raw = read_raw_egi(egi_fname, include=None)
    assert 'RawEGI' in repr(raw)
    data_read, t_read = raw[:256]
    assert_allclose(t_read, t)
    assert_allclose(data_read, data, atol=1e-10)

    include = ['TRSP', 'XXX1']
    raw = _test_raw_reader(read_raw_egi, input_fname=egi_fname,
                           include=include)

    assert_equal('eeg' in raw, True)

    eeg_chan = [c for c in raw.ch_names if c.startswith('E')]
    assert_equal(len(eeg_chan), 256)
    picks = pick_types(raw.info, eeg=True)
    assert_equal(len(picks), 256)
    assert_equal('STI 014' in raw.ch_names, True)

    events = find_events(raw, stim_channel='STI 014')
    assert_equal(len(events), 2)  # ground truth
    assert_equal(np.unique(events[:, 1])[0], 0)
    assert (np.unique(events[:, 0])[0] != 0)
    assert (np.unique(events[:, 2])[0] != 0)
    triggers = np.array([[0, 1, 1, 0], [0, 0, 1, 0]])

    # test trigger functionality
    triggers = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])
    events_ids = [12, 24]
    new_trigger = _combine_triggers(triggers, events_ids)
    assert_array_equal(np.unique(new_trigger), np.unique([0, 12, 24]))

    pytest.raises(ValueError, read_raw_egi, egi_fname, include=['Foo'],
                  preload=False)
    pytest.raises(ValueError, read_raw_egi, egi_fname, exclude=['Bar'],
                  preload=False)
    for ii, k in enumerate(include, 1):
        assert (k in raw.event_id)
        assert (raw.event_id[k] == ii)


@requires_testing_data
def test_io_egi_pns_mff():
    """Test importing EGI MFF with PNS data."""
    egi_fname_mff = op.join(data_path(), 'EGI', 'test_egi_pns.mff')
    raw = read_raw_egi(egi_fname_mff, include=None, preload=True,
                       verbose='error')
    assert ('RawMff' in repr(raw))
    pns_chans = pick_types(raw.info, ecg=True, bio=True, emg=True)
    assert_equal(len(pns_chans), 7)
    names = [raw.ch_names[x] for x in pns_chans]
    pns_names = ['Resp. Temperature'[:15],
                 'Resp. Pressure',
                 'ECG',
                 'Body Position',
                 'Resp. Effort Chest'[:15],
                 'Resp. Effort Abdomen'[:15],
                 'EMG-Leg']
    _test_raw_reader(read_raw_egi, input_fname=egi_fname_mff,
                     channel_naming='EEG %03d', verbose='error')
    assert_equal(names, pns_names)
    mat_names = [
        'Resp_Temperature'[:15],
        'Resp_Pressure',
        'ECG',
        'Body_Position',
        'Resp_Effort_Chest'[:15],
        'Resp_Effort_Abdomen'[:15],
        'EMGLeg'

    ]
    egi_fname_mat = op.join(data_path(), 'EGI', 'test_egi_pns.mat')
    mc = sio.loadmat(egi_fname_mat)
    for ch_name, ch_idx, mat_name in zip(pns_names, pns_chans, mat_names):
        print('Testing {}'.format(ch_name))
        mc_key = [x for x in mc.keys() if mat_name in x][0]
        cal = raw.info['chs'][ch_idx]['cal']
        mat_data = mc[mc_key] * cal
        raw_data = raw[ch_idx][0]
        assert_array_equal(mat_data, raw_data)


@requires_testing_data
def test_io_egi_pns_mff_bug():
    """Test importing EGI MFF with PNS data (BUG)."""
    egi_fname_mff = op.join(data_path(), 'EGI', 'test_egi_pns_bug.mff')
    with pytest.warns(RuntimeWarning, match='EGI PSG sample bug'):
        raw = read_raw_egi(egi_fname_mff, include=None, preload=True,
                           verbose='warning')
    egi_fname_mat = op.join(data_path(), 'EGI', 'test_egi_pns.mat')
    mc = sio.loadmat(egi_fname_mat)
    pns_chans = pick_types(raw.info, ecg=True, bio=True, emg=True)
    pns_names = ['Resp. Temperature'[:15],
                 'Resp. Pressure',
                 'ECG',
                 'Body Position',
                 'Resp. Effort Chest'[:15],
                 'Resp. Effort Abdomen'[:15],
                 'EMG-Leg']
    mat_names = [
        'Resp_Temperature'[:15],
        'Resp_Pressure',
        'ECG',
        'Body_Position',
        'Resp_Effort_Chest'[:15],
        'Resp_Effort_Abdomen'[:15],
        'EMGLeg'

    ]
    for ch_name, ch_idx, mat_name in zip(pns_names, pns_chans, mat_names):
        print('Testing {}'.format(ch_name))
        mc_key = [x for x in mc.keys() if mat_name in x][0]
        cal = raw.info['chs'][ch_idx]['cal']
        mat_data = mc[mc_key] * cal
        mat_data[:, -1] = 0  # The MFF has one less sample, the last one
        raw_data = raw[ch_idx][0]
        assert_array_equal(mat_data, raw_data)


@requires_testing_data
def test_io_egi_crop_no_preload():
    """Test crop non-preloaded EGI MFF data (BUG)."""
    egi_fname_mff = op.join(data_path(), 'EGI', 'test_egi.mff')
    raw = read_raw_egi(egi_fname_mff, preload=False)
    raw.crop(17.5, 20.5)
    raw.load_data()
    raw_preload = read_raw_egi(egi_fname_mff, preload=True)
    raw_preload.crop(17.5, 20.5)
    raw_preload.load_data()
    assert_allclose(raw._data, raw_preload._data)


run_tests_if_main()
