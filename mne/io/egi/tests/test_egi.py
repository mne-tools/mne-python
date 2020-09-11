# -*- coding: utf-8 -*-
# Authors: Denis A. Engemann  <denis.engemann@gmail.com>
#          simplified BSD-3 license


import os.path as op
import os
import shutil

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

base_dir = op.join(op.dirname(op.abspath(__file__)), 'data')
egi_fname = op.join(base_dir, 'test_egi.raw')
egi_txt_fname = op.join(base_dir, 'test_egi.txt')
egi_path = op.join(data_path(download=False), 'EGI')
egi_mff_fname = op.join(egi_path, 'test_egi.mff')
egi_mff_pns_fname = op.join(data_path(), 'EGI', 'test_egi_pns.mff')
egi_pause_fname = op.join(egi_path, 'test_egi_multiepoch_paused.mff')
egi_eprime_pause_fname = op.join(egi_path, 'test_egi_multiepoch_eprime.mff')
egi_pause_w1337_fname = op.join(egi_path, 'w1337_20191014_105416.mff')

# absolute event times from NetStation
egi_pause_events = {'AM40': [7.224, 11.928, 14.413, 16.848],
                    'bgin': [6.121, 8.434, 13.369, 15.815, 18.094],
                    'FIX+': [6.225, 10.929, 13.414, 15.849],
                    'ITI+': [8.293, 12.997, 15.482, 17.918]}
# absolute epoch times
egi_pause_skips = [(1304000.0, 1772000.0), (8660000.0, 12296000.0)]

egi_eprime_pause_events = {'AM40': [6.049, 8.434, 10.936, 13.321],
                           'bgin': [4.902, 7.381, 9.901, 12.268, 14.619],
                           'FIX+': [5.050, 7.435, 9.937, 12.322],
                           'ITI+': [7.185, 9.503, 12.005, 14.391]}
egi_eprime_pause_skips = [(1344000.0, 1804000.0)]

egi_pause_w1337_events = None
egi_pause_w1337_skips = [(21956000.0, 40444000.0), (60936000.0, 89332000.0)]


@requires_testing_data
@pytest.mark.parametrize('fname, skip_times, event_times', [
    (egi_pause_fname, egi_pause_skips, egi_pause_events),
    (egi_eprime_pause_fname, egi_eprime_pause_skips, egi_eprime_pause_events),
    (egi_pause_w1337_fname, egi_pause_w1337_skips, egi_pause_w1337_events),
])
def test_egi_mff_pause(fname, skip_times, event_times):
    """Test EGI MFF with pauses."""
    with pytest.warns(RuntimeWarning, match='Acquisition skips detected'):
        raw = _test_raw_reader(read_raw_egi, input_fname=fname,
                               test_scaling=False,  # XXX probably some bug
                               test_rank='less',
                               )
    assert raw.info['sfreq'] == 250.  # true for all of these files
    assert len(raw.annotations) == len(skip_times)

    # assert event onsets match expected times
    if event_times is None:
        with pytest.raises(ValueError, match='Consider using .*events_from'):
            find_events(raw)
    else:
        events = find_events(raw)
        for event_type in event_times.keys():
            ns_samples = np.floor(np.array(event_times[event_type]) *
                                  raw.info['sfreq'])
            assert_array_equal(
                events[events[:, 2] == raw.event_id[event_type], 0],
                ns_samples)

    # read some data from the middle of the skip, assert it's all zeros
    stim_picks = pick_types(raw.info, meg=False, stim=True, exclude=())
    other_picks = np.setdiff1d(np.arange(len(raw.ch_names)), stim_picks)
    for ii, annot in enumerate(raw.annotations):
        assert annot['description'] == 'BAD_ACQ_SKIP'
        start, stop = raw.time_as_index(
            [annot['onset'], annot['onset'] + annot['duration']])
        data, _ = raw[:, start:stop]
        assert_array_equal(data[other_picks], 0.)
        if event_times is not None:
            assert raw.ch_names[-1] == 'STI 014'
            assert not np.array_equal(data[stim_picks], 0.)

        # assert skips match expected onset and duration
        skip = ((start + 1) / raw.info['sfreq'] * 1e6,
                (stop + 1) / raw.info['sfreq'] * 1e6)
        assert skip == skip_times[ii]


@requires_testing_data
def test_io_egi_mff():
    """Test importing EGI MFF simple binary files."""
    raw = read_raw_egi(egi_mff_fname, include=None)
    assert ('RawMff' in repr(raw))
    include = ['DIN1', 'DIN2', 'DIN3', 'DIN4', 'DIN5', 'DIN7']
    raw = _test_raw_reader(read_raw_egi, input_fname=egi_mff_fname,
                           include=include, channel_naming='EEG %03d',
                           test_scaling=False,  # XXX probably some bug
                           )
    assert raw.info['sfreq'] == 1000.

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

    pytest.raises(ValueError, read_raw_egi, egi_mff_fname, include=['Foo'],
                  preload=False)
    pytest.raises(ValueError, read_raw_egi, egi_mff_fname, exclude=['Bar'],
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
                           include=include, test_rank='less',
                           test_scaling=False,  # XXX probably some bug
                           )

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
def test_io_egi_pns_mff(tmpdir):
    """Test importing EGI MFF with PNS data."""
    raw = read_raw_egi(egi_mff_pns_fname, include=None, preload=True,
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
    _test_raw_reader(read_raw_egi, input_fname=egi_mff_pns_fname,
                     channel_naming='EEG %03d', verbose='error',
                     test_rank='less',
                     test_scaling=False,  # XXX probably some bug
                     )
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

    # EEG missing
    new_mff = str(tmpdir.join('temp.mff'))
    shutil.copytree(egi_mff_pns_fname, new_mff)
    read_raw_egi(new_mff, verbose='error')
    os.remove(op.join(new_mff, 'info1.xml'))
    os.remove(op.join(new_mff, 'signal1.bin'))
    with pytest.raises(FileNotFoundError, match='Could not find any EEG'):
        read_raw_egi(new_mff, verbose='error')


@requires_testing_data
@pytest.mark.parametrize('preload', (True, False))
def test_io_egi_pns_mff_bug(preload):
    """Test importing EGI MFF with PNS data (BUG)."""
    egi_fname_mff = op.join(data_path(), 'EGI', 'test_egi_pns_bug.mff')
    with pytest.warns(RuntimeWarning, match='EGI PSG sample bug'):
        raw = read_raw_egi(egi_fname_mff, include=None, preload=preload,
                           verbose='warning')
    assert len(raw.annotations) == 1
    assert_allclose(raw.annotations.duration, [0.004])
    assert_allclose(raw.annotations.onset, [13.948])
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
