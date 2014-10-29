# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import warnings
import os.path as op

from nose.tools import assert_equal, assert_true
import numpy as np

from mne import io, Epochs, read_events, pick_types
from mne.utils import requires_sklearn
from mne.decoding import time_generalization
from mne.decoding import GeneralizationAcrossTime


data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)
event_id_gen = dict(aud_l=2, vis_l=4)


@requires_sklearn
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

    with warnings.catch_warnings(record=True):
        epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), preload=True, decim=decim)

        epochs_list = [epochs[k] for k in event_id.keys()]
        scores = time_generalization(epochs_list, cv=2, random_state=42)
        n_times = len(epochs.times)
        assert_true(scores.shape == (n_times, n_times))
        assert_true(scores.max() <= 1.)
        assert_true(scores.min() >= 0.)


@requires_sklearn
def test_generalization_across_time():
    """Test time generalization decoding
    """
    from sklearn.svm import SVC

    raw = io.Raw(raw_fname, preload=False)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg='mag', stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[1:13:3]
    decim = 30

    # Test on time generalization within one condition
    with warnings.catch_warnings(record=True) as w:
        epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), preload=True, decim=decim)

    # Test default running
    gat = GeneralizationAcrossTime()
    gat.fit(epochs)
    gat.predict(epochs)
    gat.score(epochs)
    gat.fit(epochs, y=epochs.events[:, 2])
    gat.score(epochs, y=epochs.events[:, 2])

    # Test basics
    # --- number of trials
    assert_true(gat.y_train_.shape[0] ==
                gat.y_true_.shape[0] ==
                gat.y_pred_.shape[2] == 14)
    # ---  number of folds
    assert_true(np.shape(gat.estimators_)[1] == gat.cv)
    # --- by default, prediction made from cv
    assert_true(not gat.independent_)
    # ---  length training size
    assert_true(len(gat.train_times['slices']) == 15 ==
                np.shape(gat.estimators_)[0])
    # ---  length testing sizes
    assert_true(len(gat.test_times_['slices']) == 15 ==
                np.shape(gat.scores_)[0])
    assert_true(len(gat.test_times_['slices'][0]) == 15
                == np.shape(gat.scores_)[1])

    # Test longer time window
    gat = GeneralizationAcrossTime(train_times={'length': .100})
    gat.fit(epochs)
    gat.score(epochs)
    assert_equal(len(gat.test_times_['slices'][0][0]), 2)
    # Decim training steps
    gat = GeneralizationAcrossTime(train_times={'step': .100})
    gat.fit(epochs)
    gat.score(epochs)
    assert_equal(len(gat.scores_), 8)

    # Test start stop training
    gat = GeneralizationAcrossTime(train_times={'start': 0.090,
                                                'stop': 0.250})
    gat.fit(epochs)
    gat.score(epochs)
    assert_equal(len(gat.scores_), 4)
    assert_equal(gat.train_times['s'][0], epochs.times[6])
    assert_equal(gat.train_times['s'][-1], epochs.times[9])

    # Test diagonal decoding
    gat = GeneralizationAcrossTime()
    gat.fit(epochs)
    gat.score(epochs, test_times='diagonal')
    assert_equal(np.shape(gat.scores_), (15, 1))

    # Test generalization across conditions
    gat = GeneralizationAcrossTime()
    gat.fit(epochs[0:6])
    gat.predict(epochs[7:], independent=True)
    gat.score(epochs[7:], independent=True)

    # Test continuous metrics
    gat = GeneralizationAcrossTime(predict_type='distance')
    gat.fit(epochs)
    gat.score(epochs)

    svc = SVC(C=1, kernel='linear', probability=True)
    gat = GeneralizationAcrossTime(clf=svc, predict_type='proba')
    gat.fit(epochs)
    gat.score(epochs)
