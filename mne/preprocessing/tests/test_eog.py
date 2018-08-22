import os.path as op

from mne import Annotations
from mne.io import read_raw_fif
from mne.preprocessing.eog import find_eog_events

data_path = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_path, 'test_raw.fif')
event_fname = op.join(data_path, 'test-eve.fif')
proj_fname = op.join(data_path, 'test-proj.fif')


def test_find_eog():
    """Test find EOG peaks."""
    raw = read_raw_fif(raw_fname)
    raw.set_annotations(Annotations([14, 21], [1, 1], 'BAD_blink'))
    events = find_eog_events(raw)
    assert len(events) == 4
    assert not all(events[:, 0] < 29000)

    events = find_eog_events(raw, reject_by_annotation=True)
    assert all(events[:, 0] < 29000)

    # threshold option
    events_thr = find_eog_events(raw, thresh=100e-6)
    assert len(events_thr) == 5
