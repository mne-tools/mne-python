# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import warnings
import os.path as op

from nose.tools import assert_true
import numpy as np

from mne import io, Epochs, read_events, pick_types
from mne.utils import requires_sklearn
from mne.decoding import time_generalization
from mne.decoding import GeneralizationAcrossTime
from mne.fixes import partial
from mne.utils import create_slices, deprecated


data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)
event_id_gen = dict(aud_l=2, vis_l=4)


@requires_sklearn
@deprecated("'test_time_generalization' will be removed in v0.9. Use"
            " 'test_GeneralizationAcrossTime' instead.")
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
        # Test on time generalization within one condition
        epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), preload=True, decim=decim)

        epochs_list = [epochs[k] for k in event_id.keys()]
        results = time_generalization(epochs_list, cv=2, random_state=42)
        scores = results['scores']
        n_slices = len(epochs.times)
        # Test that by default, the temporal generalization is trained and
        # Tested across all time points
        assert_true(scores.shape == (n_slices, n_slices))
        # Test that the decoding scores are between 0 and 1
        assert_true(scores.max() <= 1.)
        assert_true(scores.min() >= 0.)
        # Test that traing and testing time are correct for asymetrical
        # training and testing times
        n_slices = len(epochs_list[0].times)
        train_slices = create_slices(0, n_slices, step=2)
        test_slices = [create_slices(0, n_slices)] * \
            len(train_slices)
        results = time_generalization(epochs_list, cv=2, random_state=42,
                                      train_slices=train_slices,
                                      test_slices=test_slices)
        scores = results['scores']
        assert_true(scores.shape == (8, 15))
        # Test create_slice callable
        train_slices = partial(create_slices, step=2)
        results = time_generalization(epochs_list, cv=2, random_state=42,
                                      train_slices=train_slices)
        # Test on time generalization within across two conditions
        epochs_list_gen = Epochs(raw, events, event_id_gen, tmin, tmax,
                                 picks=picks, baseline=(None, 0),
                                 preload=True, decim=decim)
        epochs_list_gen = [epochs_list_gen[k] for k in event_id.keys()]
        results = time_generalization(epochs_list,
                                      epochs_list_gen=epochs_list_gen,
                                      cv=2, random_state=42)
        scores = results['scores']
        scores_gen = results['scores_gen']
        assert_true(scores.shape == scores_gen.shape)
        assert_true(scores_gen.max() <= 1.)
        assert_true(scores_gen.min() >= 0.)

        # Test parallelization
        time_generalization(epochs_list, parallel_across='time_samples')
        time_generalization(epochs_list, parallel_across='folds')


def test_GeneralizationAcrossTime():
    """Test time generalization decoding
    """
    raw = io.Raw(raw_fname, preload=False)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg='mag', stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[1:13:3]
    decim = 30

    with warnings.catch_warnings(record=True) as w:
        # Test on time generalization within one condition
        epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), preload=True, decim=decim)

        # Test default running
        gat = GeneralizationAcrossTime()
        gat.fit(epochs)
        gat.predict(epochs)
        gat.score(epochs)
        gat.fit(epochs, y=epochs.events[:,2])
        gat.score(epochs, y=epochs.events[:,2])

        # Test basics
        # --- number of trials
        assert_true(gat.y_train_.shape[0] ==
                    gat.y_true_.shape[0] ==
                    gat.y_pred_.shape[2] == 14)
        # ---  number of folds
        assert_true(np.shape(gat.estimators_)[1] == gat.cv_.n_folds)
        # --- by default, prediction made from cv
        assert_true(not gat.independent_)
        # ---  length training size
        assert_true(
            len(gat.train_times_['slices']) == 15 == np.shape(gat.estimators_)[0])
        # ---  length testing sizes
        assert_true(
            len(gat.test_times_['slices']) == 15 == np.shape(gat.scores_)[0])
        assert_true(
            len(gat.test_times_['slices'][0]) == 15 == np.shape(gat.scores_)[1])

        # Test longer time window
        gat = GeneralizationAcrossTime(train_times={'length': .100})
        gat.fit(epochs)
        gat.score(epochs)
        assert(len(gat.test_times_['slices'][0][0])==2)
        # Decim training steps
        gat = GeneralizationAcrossTime(train_times={'step': .100})
        gat.fit(epochs)
        gat.score(epochs)
        assert(len(gat.scores_)==8)

        # Test start stop training
        gat = GeneralizationAcrossTime(
            train_times={'start': 0.090, 'stop': 0.250})
        gat.fit(epochs)
        gat.score(epochs)
        assert(len(gat.scores_)==4)
        assert(gat.train_times_['s'][0]==epochs.times[6])
        assert(gat.train_times_['s'][-1]==epochs.times[9])

        # Test diagonal decoding
        gat = GeneralizationAcrossTime()
        gat.fit(epochs)
        gat.score(epochs, test_times='diagonal')
        assert(np.shape(gat.scores_)==(15, 1))

        # Test generalization across conditions
        gat = GeneralizationAcrossTime()
        gat.fit(epochs[0:6])
        gat.predict(epochs[7:], independent=True)
        gat.score(epochs[7:], independent=True)

        # Test continuous metrics
        gat = GeneralizationAcrossTime()
        gat.fit(epochs)
        gat.score(epochs, predict_type='distance')
        from sklearn.svm import SVC
        svc = SVC(C=1, kernel='linear', probability=True)
        gat = GeneralizationAcrossTime(clf=svc)
        gat.fit(epochs)
        gat.score(epochs, predict_type='proba')

        # XXX Test parallelization ?
