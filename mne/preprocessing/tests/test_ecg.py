import os.path as op
import pytest
import numpy as np

from mne.io import read_raw_fif
from mne import pick_types
from mne.preprocessing import find_ecg_events, create_ecg_epochs

data_path = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_path, 'test_raw.fif')
event_fname = op.join(data_path, 'test-eve.fif')
proj_fname = op.join(data_path, 'test-proj.fif')


def test_find_ecg():
    """Test find ECG peaks."""
    # Test if ECG analysis will work on data that is not preloaded
    raw = read_raw_fif(raw_fname, preload=False).pick_types(meg=True)
    raw.pick(raw.ch_names[::10] + ['MEG 2641'])
    raw.info.normalize_proj()

    # once with mag-trick
    # once with characteristic channel
    raw_bad = raw.copy().load_data()
    ecg_idx = raw.ch_names.index('MEG 1531')
    raw_bad._data[ecg_idx, :1] = 1e6  # this will break the detector
    raw_bad.annotations.append(raw.first_samp / raw.info['sfreq'],
                               1. / raw.info['sfreq'], 'BAD_values')
    raw_noload = raw.copy()
    raw.resample(100)

    for ch_name, tstart in zip(['MEG 1531', None],
                               [raw.times[-1] / 2, 0]):
        events, ch_ECG, average_pulse, ecg = find_ecg_events(
            raw, event_id=999, ch_name=ch_name, tstart=tstart,
            return_ecg=True)
        assert raw.n_times == ecg.shape[-1]
        assert 40 < average_pulse < 60
        n_events = len(events)

        # with annotations
        average_pulse = find_ecg_events(raw_bad, ch_name=ch_name,
                                        tstart=tstart,
                                        reject_by_annotation=False)[2]
        assert average_pulse < 1.
        average_pulse = find_ecg_events(raw_bad, ch_name=ch_name,
                                        tstart=tstart,
                                        reject_by_annotation=True)[2]
        assert 55 < average_pulse < 60

    picks = pick_types(
        raw.info, meg='grad', eeg=False, stim=False,
        eog=False, ecg=True, emg=False, ref_meg=False,
        exclude='bads')

    # There should be no ECG channels, or else preloading will not be
    # tested
    assert 'ecg' not in raw

    ecg_epochs = create_ecg_epochs(raw_noload, picks=picks, keep_ecg=True)
    assert len(ecg_epochs.events) == n_events
    assert 'ECG-SYN' not in raw.ch_names
    assert 'ECG-SYN' in ecg_epochs.ch_names
    assert len(ecg_epochs) == 23

    picks = pick_types(
        ecg_epochs.info, meg=False, eeg=False, stim=False,
        eog=False, ecg=True, emg=False, ref_meg=False,
        exclude='bads')
    assert len(picks) == 1

    ecg_epochs = create_ecg_epochs(raw, ch_name='MEG 2641')
    assert 'MEG 2641' in ecg_epochs.ch_names

    # test with user provided ecg channel
    raw.del_proj()
    assert 'MEG 2641' in raw.ch_names
    with pytest.warns(RuntimeWarning, match='unit for channel'):
        raw.set_channel_types({'MEG 2641': 'ecg'})
    create_ecg_epochs(raw)

    raw.pick_types(meg=True)  # remove ECG
    assert 'MEG 2641' not in raw.ch_names
    ecg_epochs = create_ecg_epochs(raw, keep_ecg=False)
    assert len(ecg_epochs.events) == n_events
    assert 'ECG-SYN' not in raw.ch_names
    assert 'ECG-SYN' not in ecg_epochs.ch_names

    # Test behavior if no peaks can be found -> achieve this by providing
    # all-zero'd data
    raw._data[ecg_idx] = 0.
    ecg_events, _, average_pulse, ecg = find_ecg_events(
        raw, ch_name=raw.ch_names[ecg_idx], return_ecg=True
    )
    assert ecg_events.size == 0
    assert average_pulse == 0
    assert np.allclose(ecg, np.zeros_like(ecg))
