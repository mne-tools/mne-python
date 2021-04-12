import os.path as op

import pytest

from mne import Annotations
from mne.io import read_raw_fif
from mne.preprocessing.eog import find_eog_events

data_path = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_path, 'test_raw.fif')
event_fname = op.join(data_path, 'test-eve.fif')
proj_fname = op.join(data_path, 'test-proj.fif')


@pytest.mark.parametrize('ch_name', (None,
                                     'EOG 061',
                                     'EEG 060,EOG 061',
                                     ['EEG 060', 'EOG 061']))
def test_find_eog(ch_name):
    """Test find EOG peaks."""
    raw = read_raw_fif(raw_fname)
    raw.set_annotations(Annotations([14, 21], [1, 1], 'BAD_blink'))

    if ch_name is not None and ',' in ch_name:
        with pytest.warns(DeprecationWarning, match='pass a list of channels'):
            events = find_eog_events(raw, ch_name=ch_name)
            return

    events = find_eog_events(raw, ch_name=ch_name)
    assert len(events) == 4
    assert not all(events[:, 0] < 29000)

    events = find_eog_events(raw, reject_by_annotation=True, ch_name=ch_name)
    assert all(events[:, 0] < 29000)

    # threshold option
    events_thr = find_eog_events(raw, thresh=100e-6, ch_name=ch_name)
    assert len(events_thr) == 5
