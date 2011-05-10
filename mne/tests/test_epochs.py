# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import os.path as op

import mne
from mne import fiff

raw_fname = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                     'test_raw.fif')
event_name = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                     'test-eve.fif')


def test_read_epochs():
    """Reading epochs from raw files
    """
    event_id = 1
    tmin = -0.2
    tmax = 0.5

    # Setup for reading the raw data
    raw = fiff.Raw(raw_fname)
    events = mne.read_events(event_name)
    picks = fiff.pick_types(raw.info, meg=True, eeg=False, stim=False,
                            eog=True, include=['STI 014'])
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0))
    epochs.average()
    data = epochs.get_data()

    eog_picks = fiff.pick_types(raw.info, meg=False, eeg=False, stim=False,
                                eog=True)
    epochs.drop_picks(eog_picks)
    data_no_eog = epochs.get_data()
    assert data.shape[1] == (data_no_eog.shape[1] + len(eog_picks))


def test_reject_epochs():
    """Test of epochs rejection
    """
    event_id = 1
    tmin = -0.2
    tmax = 0.5

    # Setup for reading the raw data
    raw = fiff.Raw(raw_fname)
    events = mne.read_events(event_name)

    picks = fiff.pick_types(raw.info, meg=True, eeg=True, stim=True,
                            eog=True, include=['STI 014'])
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0),
                        reject=dict(grad=1000e-12, mag=4e-12, eeg=80e-6,
                                    eog=150e-6),
                        flat=dict(grad=1e-15, mag=1e-15))
    data = epochs.get_data()
    n_events = len(epochs.events)
    n_clean_epochs = len(data)
    # Should match
    # mne_process_raw --raw test_raw.fif --projoff \
    #   --saveavetag -ave --ave test.ave --filteroff
    assert n_events > n_clean_epochs
    assert n_clean_epochs == 3
