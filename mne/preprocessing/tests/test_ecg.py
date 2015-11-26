import os.path as op

from nose.tools import assert_true, assert_equal

from mne.io import Raw
from mne import pick_types
from mne.preprocessing.ecg import find_ecg_events, create_ecg_epochs

data_path = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_path, 'test_raw.fif')
event_fname = op.join(data_path, 'test-eve.fif')
proj_fname = op.join(data_path, 'test-proj.fif')


def test_find_ecg():
    """Test find ECG peaks"""
    raw = Raw(raw_fname)

    # once with mag-trick
    # once with characteristic channel
    for ch_name in ['MEG 1531', None]:
        events, ch_ECG, average_pulse, ecg = find_ecg_events(
            raw, event_id=999, ch_name=None, return_ecg=True)
        assert_equal(len(raw.times), len(ecg))
        n_events = len(events)
        _, times = raw[0, :]
        assert_true(55 < average_pulse < 60)

    picks = pick_types(
        raw.info, meg='grad', eeg=False, stim=False,
        eog=False, ecg=True, emg=False, ref_meg=False,
        exclude='bads')

    raw.load_data()
    ecg_epochs = create_ecg_epochs(raw, picks=picks, keep_ecg=True)
    assert_equal(len(ecg_epochs.events), n_events)
    assert_true('ECG-SYN' not in raw.ch_names)
    assert_true('ECG-SYN' in ecg_epochs.ch_names)

    picks = pick_types(
        ecg_epochs.info, meg=False, eeg=False, stim=False,
        eog=False, ecg=True, emg=False, ref_meg=False,
        exclude='bads')
    assert_true(len(picks) == 1)
