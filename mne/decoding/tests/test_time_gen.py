# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import warnings
import os.path as op

from nose.tools import assert_true

from mne import io, Epochs, read_events, pick_types
from mne.utils import _TempDir, requires_sklearn
from mne.decoding import time_generalization

tempdir = _TempDir()

data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)
event_id_gen = dict(aud_l=2, vis_l=4)


@requires_sklearn
def test_time_generalization():
    """Test time generalization decoding
    """
    raw = io.Raw(raw_fname, preload=False)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg='mag', stim=False, ecg=False,
                            eog=False, exclude='bads')
    picks = picks[1:13:3]
    decim = 30

    with warnings.catch_warnings(record=True) as w:
        # test on time generalization within one condition
        epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), preload=True, decim=decim)

        epochs_list = [epochs[k] for k in event_id.keys()]
        scores = time_generalization(epochs_list, cv=2, random_state=42)
        n_times = len(epochs.times)
        assert_true(scores.shape == (n_times, n_times))
        assert_true(scores.max() <= 1.)
        assert_true(scores.min() >= 0.)
        # test on time generalization within across two conditions
        epochs_list_generalize = Epochs(raw, events, event_id_gen, tmin, tmax, picks=picks,
                        baseline=(None, 0), preload=True, decim=decim)
        epochs_list_generalize = [epochs_list_generalize[k] for k in event_id.keys()]
        scores, scores_generalize = time_generalization(epochs_list, 
            epochs_list_generalize=epochs_list_generalize,
            cv=2, random_state=42)
        assert_true(scores.shape == scores_generalize.shape)
        assert_true(scores_generalize.max() <= 1.)
        assert_true(scores_generalize.min() >= 0.)