import os.path as op

from nose.tools import assert_true, assert_equal

from mne.io import Raw
from mne.preprocessing.ecg import find_ecg_events, create_ecg_epochs

data_path = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_path, 'test_raw.fif')
event_fname = op.join(data_path, 'test-eve.fif')
proj_fname = op.join(data_path, 'test-proj.fif')


def test_find_ecg():
    """Test find ECG peaks"""
    raw = Raw(raw_fname)
    events, ch_ECG, average_pulse = find_ecg_events(raw, event_id=999,
                                                    ch_name='MEG 1531')
    n_events = len(events)
    _, times = raw[0, :]
    assert_true(55 < average_pulse < 60)

    ecg_epochs = create_ecg_epochs(raw, ch_name='MEG 1531')
    assert_equal(len(ecg_epochs.events), n_events)
