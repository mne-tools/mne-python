import os.path as op

import pytest
from mne.io import read_raw_fif
from mne import pick_types
from mne.preprocessing import find_ecg_events, create_ecg_epochs
from mne.utils import run_tests_if_main

data_path = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_path, 'test_raw.fif')
event_fname = op.join(data_path, 'test-eve.fif')
proj_fname = op.join(data_path, 'test-proj.fif')


def test_find_ecg():
    """Test find ECG peaks."""
    # Test if ECG analysis will work on data that is not preloaded
    raw = read_raw_fif(raw_fname, preload=False)

    # once with mag-trick
    # once with characteristic channel
    raw_bad = raw.copy().load_data()
    ecg_idx = raw.ch_names.index('MEG 1531')
    raw_bad._data[ecg_idx, :1] = 1e6  # this will break the detector
    raw_bad.annotations.append(raw.first_samp / raw.info['sfreq'],
                               1. / raw.info['sfreq'], 'BAD_values')
    for ch_name in ['MEG 1531', None]:
        events, ch_ECG, average_pulse, ecg = find_ecg_events(
            raw, event_id=999, ch_name=ch_name, return_ecg=True)
        assert raw.n_times == ecg.shape[-1]
        n_events = len(events)
        _, times = raw[0, :]
        assert 55 < average_pulse < 60
        # with annotations
        average_pulse = find_ecg_events(raw_bad, ch_name=ch_name,
                                        reject_by_annotation=False)[2]
        assert average_pulse < 1.
        average_pulse = find_ecg_events(raw_bad, ch_name=ch_name,
                                        reject_by_annotation=True)[2]
        assert 55 < average_pulse < 60
    average_pulse = find_ecg_events(raw_bad, ch_name='MEG 2641',
                                    reject_by_annotation=False)[2]
    assert 55 < average_pulse < 65
    del raw_bad

    picks = pick_types(
        raw.info, meg='grad', eeg=False, stim=False,
        eog=False, ecg=True, emg=False, ref_meg=False,
        exclude='bads')

    # There should be no ECG channels, or else preloading will not be
    # tested
    assert 'ecg' not in raw

    ecg_epochs = create_ecg_epochs(raw, picks=picks, keep_ecg=True)
    assert len(ecg_epochs.events) == n_events
    assert 'ECG-SYN' not in raw.ch_names
    assert 'ECG-SYN' in ecg_epochs.ch_names

    picks = pick_types(
        ecg_epochs.info, meg=False, eeg=False, stim=False,
        eog=False, ecg=True, emg=False, ref_meg=False,
        exclude='bads')
    assert len(picks) == 1

    ecg_epochs = create_ecg_epochs(raw, ch_name='MEG 2641')
    assert 'MEG 2641' in ecg_epochs.ch_names

    # test with user provided ecg channel
    raw.info['projs'] = list()
    with pytest.warns(RuntimeWarning, match='unit for channel'):
        raw.set_channel_types({'MEG 2641': 'ecg'})
    create_ecg_epochs(raw)

    raw.load_data().pick_types()  # remove ECG
    ecg_epochs = create_ecg_epochs(raw, keep_ecg=False)
    assert len(ecg_epochs.events) == n_events
    assert 'ECG-SYN' not in raw.ch_names
    assert 'ECG-SYN' not in ecg_epochs.ch_names


run_tests_if_main()
