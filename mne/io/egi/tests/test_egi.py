# -*- coding: utf-8 -*-
# Authors: Denis A. Engemann  <denis.engemann@gmail.com>
#          simplified BSD-3 license


from pathlib import Path
import os.path as op
import os
import shutil

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pytest
from scipy import io as sio

from mne import find_events, pick_types
from mne.io import read_raw_egi, read_evokeds_mff, read_raw_fif
from mne.io.constants import FIFF
from mne.io.egi.egi import _combine_triggers
from mne.io.tests.test_raw import _test_raw_reader
from mne.utils import requires_version, object_diff
from mne.datasets.testing import data_path, requires_testing_data

base_dir = op.join(op.dirname(op.abspath(__file__)), 'data')
egi_fname = op.join(base_dir, 'test_egi.raw')
egi_txt_fname = op.join(base_dir, 'test_egi.txt')
egi_path = op.join(data_path(download=False), 'EGI')
egi_mff_fname = op.join(egi_path, 'test_egi.mff')
egi_mff_pns_fname = op.join(egi_path, 'test_egi_pns.mff')
egi_pause_fname = op.join(egi_path, 'test_egi_multiepoch_paused.mff')
egi_eprime_pause_fname = op.join(egi_path, 'test_egi_multiepoch_eprime.mff')
egi_pause_w1337_fname = op.join(egi_path, 'w1337_20191014_105416.mff')
egi_mff_evoked_fname = op.join(egi_path, 'test_egi_evoked.mff')
egi_txt_evoked_cat1_fname = op.join(egi_path, 'test_egi_evoked_cat1.txt')
egi_txt_evoked_cat2_fname = op.join(egi_path, 'test_egi_evoked_cat2.txt')

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
    if fname == egi_pause_w1337_fname:
        # too slow to _test_raw_reader
        raw = read_raw_egi(fname).load_data()
    else:
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
@pytest.mark.parametrize('fname', [
    egi_pause_fname,
    egi_eprime_pause_fname,
    egi_pause_w1337_fname,
])
def test_egi_mff_pause_chunks(fname, tmp_path):
    """Test that on-demand of all short segments works (via I/O)."""
    fname_temp = tmp_path / 'test_raw.fif'
    raw_data = read_raw_egi(fname, preload=True).get_data()
    raw = read_raw_egi(fname)
    with pytest.warns(RuntimeWarning, match='Acquisition skips detected'):
        raw.save(fname_temp)
    del raw
    raw_data_2 = read_raw_fif(fname_temp).get_data()
    assert_allclose(raw_data, raw_data_2)


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
    # The ref here is redundant, but we don't currently have a way in
    # DigMontage to mark that a given channel is actually the ref so...
    assert len(raw.info['dig']) == 133  # 129 eeg + 1 ref + 3 cardinal points
    assert raw.info['dig'][0]['ident'] == 1  # EEG channel E1
    assert raw.info['dig'][3]['ident'] == 0  # Reference channel
    assert raw.info['dig'][-1]['ident'] == 129  # Reference channel
    ref_loc = raw.info['dig'][3]['r']
    eeg_picks = pick_types(raw.info, eeg=True)
    assert len(eeg_picks) == 129
    for i in eeg_picks:
        loc = raw.info['chs'][i]['loc']
        assert loc[:3].any(), loc[:3]
        assert_array_equal(loc[3:6], ref_loc, err_msg=f'{i}')
    assert raw.info['device_info']['type'] == 'HydroCel GSN 128 1.0'

    assert 'eeg' in raw
    eeg_chan = [c for c in raw.ch_names if 'EEG' in c]
    assert len(eeg_chan) == 129
    assert 'STI 014' in raw.ch_names

    events = find_events(raw, stim_channel='STI 014')
    assert len(events) == 8
    assert np.unique(events[:, 1])[0] == 0
    assert np.unique(events[:, 0])[0] != 0
    assert np.unique(events[:, 2])[0] != 0

    with pytest.raises(ValueError, match='Could not find event'):
        read_raw_egi(egi_mff_fname, include=['Foo'])
    with pytest.raises(ValueError, match='Could not find event'):
        read_raw_egi(egi_mff_fname, exclude=['Bar'])
    for ii, k in enumerate(include, 1):
        assert k in raw.event_id
        assert raw.event_id[k] == ii


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

    # The reader should accept a Path, too.
    with pytest.warns(RuntimeWarning, match='Did not find any event code'):
        raw = read_raw_egi(Path(egi_fname), include=None)

    assert 'RawEGI' in repr(raw)
    data_read, t_read = raw[:256]
    assert_allclose(t_read, t)
    assert_allclose(data_read, data, atol=1e-10)

    include = ['TRSP', 'XXX1']
    raw = _test_raw_reader(read_raw_egi, input_fname=egi_fname,
                           include=include, test_rank='less',
                           test_scaling=False,  # XXX probably some bug
                           )

    assert 'eeg' in raw

    eeg_chan = [c for c in raw.ch_names if c.startswith('E')]
    assert len(eeg_chan) == 256
    picks = pick_types(raw.info, eeg=True)
    assert len(picks) == 256
    assert 'STI 014' in raw.ch_names

    events = find_events(raw, stim_channel='STI 014')
    assert len(events) == 2  # ground truth
    assert np.unique(events[:, 1])[0] == 0
    assert np.unique(events[:, 0])[0] != 0
    assert np.unique(events[:, 2])[0] != 0
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
def test_io_egi_pns_mff(tmp_path):
    """Test importing EGI MFF with PNS data."""
    raw = read_raw_egi(egi_mff_pns_fname, include=None, preload=True,
                       verbose='error')
    assert ('RawMff' in repr(raw))
    pns_chans = pick_types(raw.info, ecg=True, bio=True, emg=True)
    assert len(pns_chans) == 7
    names = [raw.ch_names[x] for x in pns_chans]
    pns_names = ['Resp. Temperature',
                 'Resp. Pressure',
                 'ECG',
                 'Body Position',
                 'Resp. Effort Chest',
                 'Resp. Effort Abdomen',
                 'EMG-Leg']
    _test_raw_reader(read_raw_egi, input_fname=egi_mff_pns_fname,
                     channel_naming='EEG %03d', verbose='error',
                     test_rank='less',
                     test_scaling=False,  # XXX probably some bug
                     )
    assert names == pns_names
    mat_names = [
        'Resp_Temperature',
        'Resp_Pressure',
        'ECG',
        'Body_Position',
        'Resp_Effort_Chest',
        'Resp_Effort_Abdomen',
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
    new_mff = tmp_path / 'temp.mff'
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


@requires_version('mffpy', '0.5.7')
@requires_testing_data
@pytest.mark.parametrize('idx, cond, tmax, signals, bads', [
    (0, 'Category 1', 0.016, egi_txt_evoked_cat1_fname,
     ['E8', 'E11', 'E17', 'E28', 'ECG']),
    (1, 'Category 2', 0.0, egi_txt_evoked_cat2_fname,
     ['E257', 'EMG'])
])
def test_io_egi_evokeds_mff(idx, cond, tmax, signals, bads):
    """Test reading evoked MFF file."""
    # Test reading all conditions from evokeds
    evokeds = read_evokeds_mff(egi_mff_evoked_fname)
    assert len(evokeds) == 2
    # Test reading list of conditions from evokeds
    evokeds = read_evokeds_mff(egi_mff_evoked_fname, condition=[0, 1])
    assert len(evokeds) == 2
    # Test invalid condition
    with pytest.raises(ValueError) as exc_info:
        read_evokeds_mff(egi_mff_evoked_fname, condition='Invalid Condition')
    message = "Invalid value for the 'condition' parameter provided as " \
              "category name. Allowed values are 'Category 1', and " \
              "'Category 2', but got 'Invalid Condition' instead."
    assert str(exc_info.value) == message
    with pytest.raises(ValueError) as exc_info:
        read_evokeds_mff(egi_mff_evoked_fname, condition=2)
    message = '"condition" parameter (2), provided as epoch index, ' \
              'is out of range for available epochs (2).'
    assert str(exc_info.value) == message
    with pytest.raises(TypeError) as exc_info:
        read_evokeds_mff(egi_mff_evoked_fname, condition=1.2)
    message = '"condition" parameter must be either int or str.'
    assert str(exc_info.value) == message
    # Test reading evoked data from single condition
    evoked_cond = read_evokeds_mff(egi_mff_evoked_fname, condition=cond)
    evoked_idx = read_evokeds_mff(egi_mff_evoked_fname, condition=idx)
    for evoked in [evoked_cond, evoked_idx]:
        assert evoked.comment == cond
        assert evoked.nave == 3
        assert evoked.tmin == 0.0
        assert evoked.tmax == tmax
    # Check signal data
    data = np.loadtxt(signals, ndmin=2).T * 1e-6  # convert to volts
    assert_allclose(evoked_cond.data, data, atol=1e-12)
    assert_allclose(evoked_idx.data, data, atol=1e-12)
    # Check info
    assert object_diff(evoked_cond.info, evoked_idx.info) == ''
    assert evoked_cond.info['description'] == cond
    assert evoked_cond.info['bads'] == bads
    assert len(evoked_cond.info['ch_names']) == 259
    assert 'ECG' in evoked_cond.info['ch_names']
    assert 'EMG' in evoked_cond.info['ch_names']
    assert 'ecg' in evoked_cond
    assert 'emg' in evoked_cond
    pick_eeg = pick_types(evoked_cond.info, eeg=True, exclude=[])
    assert len(pick_eeg) == 257
    assert evoked_cond.info['nchan'] == 259
    assert evoked_cond.info['sfreq'] == 250.0
    assert not evoked_cond.info['custom_ref_applied']
    assert len(evoked_cond.info['dig']) == 261
    assert evoked_cond.info['device_info']['type'] == 'HydroCel GSN 256 1.0'


@requires_version('mffpy', '0.5.7')
@requires_testing_data
def test_read_evokeds_mff_bad_input():
    """Test errors are thrown when reading invalid input file."""
    # Test file that is not an MFF
    with pytest.raises(ValueError) as exc_info:
        read_evokeds_mff(egi_fname)
    message = 'fname must be an MFF file with extension ".mff".'
    assert str(exc_info.value) == message
    # Test continuous MFF
    with pytest.raises(ValueError) as exc_info:
        read_evokeds_mff(egi_mff_fname)
    message = f'{egi_mff_fname} is a continuous MFF file. ' \
              'fname must be the path to an averaged MFF file.'
    assert str(exc_info.value) == message


@requires_testing_data
def test_egi_coord_frame():
    """Test that EGI coordinate frame is changed to head."""
    info = read_raw_egi(egi_mff_fname).info
    want_idents = (
        FIFF.FIFFV_POINT_LPA,
        FIFF.FIFFV_POINT_NASION,
        FIFF.FIFFV_POINT_RPA,
    )
    for ii, want in enumerate(want_idents):
        d = info['dig'][ii]
        assert d['kind'] == FIFF.FIFFV_POINT_CARDINAL
        assert d['ident'] == want
        loc = d['r']
        if ii == 0:
            assert 0.05 < -loc[0] < 0.1, 'LPA'
            assert_allclose(loc[1:], 0, atol=1e-7, err_msg='LPA')
        elif ii == 1:
            assert 0.05 < loc[1] < 0.11, 'Nasion'
            assert_allclose(loc[::2], 0, atol=1e-7, err_msg='Nasion')
        else:
            assert ii == 2
            assert 0.05 < loc[0] < 0.1, 'RPA'
            assert_allclose(loc[1:], 0, atol=1e-7, err_msg='RPA')
    for d in info['dig'][3:]:
        assert d['kind'] == FIFF.FIFFV_POINT_EEG
