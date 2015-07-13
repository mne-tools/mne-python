# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import warnings
import copy
import os.path as op

from nose.tools import assert_equal, assert_true, assert_raises
import numpy as np
from numpy.testing import assert_array_equal

from mne import io, Epochs, read_events, pick_types
from mne.utils import requires_sklearn, slow_test
from mne.decoding import GeneralizationAcrossTime


data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)
event_id_gen = dict(aud_l=2, vis_l=4)


@slow_test
@requires_sklearn
def test_generalization_across_time():
    """Test time generalization decoding
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_squared_error

    raw = io.Raw(raw_fname, preload=False)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg='mag', stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[0:2]
    decim = 30

    # Test on time generalization within one condition
    with warnings.catch_warnings(record=True):
        epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), preload=True, decim=decim)
    # Test default running
    gat = GeneralizationAcrossTime(picks='foo')
    assert_equal("<GAT | no fit, no prediction, no score>", "%s" % gat)
    assert_raises(ValueError, gat.fit, epochs)
    with warnings.catch_warnings(record=True):
        # check classic fit + check manual picks
        gat.picks = [0]
        gat.fit(epochs)
        # check optional y as array
        gat.picks = None
        gat.fit(epochs, y=epochs.events[:, 2])
        # check optional y as list
        gat.fit(epochs, y=epochs.events[:, 2].tolist())
    assert_equal(len(gat.picks_), len(gat.ch_names), 1)
    assert_equal("<GAT | fitted, start : -0.200 (s), stop : 0.499 (s), no "
                 "prediction, no score>", '%s' % gat)
    assert_equal(gat.ch_names, epochs.ch_names)
    gat.predict(epochs)
    assert_equal("<GAT | fitted, start : -0.200 (s), stop : 0.499 (s), "
                 "predicted 14 epochs, no score>",
                 "%s" % gat)
    gat.score(epochs)
    gat.score(epochs, y=epochs.events[:, 2])
    gat.score(epochs, y=epochs.events[:, 2].tolist())
    assert_equal("<GAT | fitted, start : -0.200 (s), stop : 0.499 (s), "
                 "predicted 14 epochs,\n scored "
                 "(accuracy_score)>", "%s" % gat)
    with warnings.catch_warnings(record=True):
        gat.fit(epochs, y=epochs.events[:, 2])

    old_mode = gat.predict_mode
    gat.predict_mode = 'super-foo-mode'
    assert_raises(ValueError, gat.predict, epochs)
    gat.predict_mode = old_mode

    gat.score(epochs, y=epochs.events[:, 2])
    assert_true("accuracy_score" in '%s' % gat.scorer_)
    epochs2 = epochs.copy()

    # check _DecodingTime class
    assert_equal("<DecodingTime | start: -0.200 (s), stop: 0.499 (s), step: "
                 "0.047 (s), length: 0.047 (s), n_time_windows: 15>",
                 "%s" % gat.train_times_)
    assert_equal("<DecodingTime | start: -0.200 (s), stop: 0.499 (s), step: "
                 "0.047 (s), length: 0.047 (s), n_time_windows: 15 x 15>",
                 "%s" % gat.test_times_)

    # the y-check
    gat.predict_mode = 'mean-prediction'
    epochs2.events[:, 2] += 10
    gat_ = copy.deepcopy(gat)
    assert_raises(ValueError, gat_.score, epochs2)
    gat.predict_mode = 'cross-validation'

    # Test basics
    # --- number of trials
    assert_true(gat.y_train_.shape[0] ==
                gat.y_true_.shape[0] ==
                len(gat.y_pred_[0][0]) == 14)
    # ---  number of folds
    assert_true(np.shape(gat.estimators_)[1] == gat.cv)
    # ---  length training size
    assert_true(len(gat.train_times_['slices']) == 15 ==
                np.shape(gat.estimators_)[0])
    # ---  length testing sizes
    assert_true(len(gat.test_times_['slices']) == 15 ==
                np.shape(gat.scores_)[0])
    assert_true(len(gat.test_times_['slices'][0]) == 15 ==
                np.shape(gat.scores_)[1])

    # Test longer time window
    gat = GeneralizationAcrossTime(train_times={'length': .100})
    with warnings.catch_warnings(record=True):
        gat2 = gat.fit(epochs)
    assert_true(gat is gat2)  # return self
    assert_true(hasattr(gat2, 'cv_'))
    assert_true(gat2.cv_ != gat.cv)
    scores = gat.score(epochs)
    assert_true(isinstance(scores, list))  # type check
    assert_equal(len(scores[0]), len(scores))  # shape check

    assert_equal(len(gat.test_times_['slices'][0][0]), 2)
    # Decim training steps
    gat = GeneralizationAcrossTime(train_times={'step': .100})
    with warnings.catch_warnings(record=True):
        gat.fit(epochs)

    gat.score(epochs)
    assert_true(len(gat.scores_) == len(gat.estimators_) == 8)  # training time
    assert_equal(len(gat.scores_[0]), 15)  # testing time

    # Test start stop training
    gat = GeneralizationAcrossTime(train_times={'start': 0.090,
                                                'stop': 0.250})
    # predict without fit
    assert_raises(RuntimeError, gat.predict, epochs)
    with warnings.catch_warnings(record=True):
        gat.fit(epochs)
    gat.score(epochs)
    assert_equal(len(gat.scores_), 4)
    assert_equal(gat.train_times_['times'][0], epochs.times[6])
    assert_equal(gat.train_times_['times'][-1], epochs.times[9])

    # Test score without passing epochs & Test diagonal decoding
    gat = GeneralizationAcrossTime(test_times='diagonal')
    with warnings.catch_warnings(record=True):
        gat.fit(epochs)
    assert_raises(RuntimeError, gat.score)
    gat.predict(epochs)
    scores = gat.score()
    assert_true(scores is gat.scores_)
    assert_equal(np.shape(gat.scores_), (15, 1))
    assert_array_equal([tim for ttime in gat.test_times_['times']
                        for tim in ttime], gat.train_times_['times'])

    # Test generalization across conditions
    gat = GeneralizationAcrossTime(predict_mode='mean-prediction')
    with warnings.catch_warnings(record=True):
        gat.fit(epochs[0:6])
    gat.predict(epochs[7:])
    gat.score(epochs[7:])

    # Test wrong testing time
    gat.test_times = 'foo'
    assert_raises(ValueError, gat.predict, epochs)
    assert_raises(RuntimeError, gat.score)

    svc = SVC(C=1, kernel='linear', probability=True)
    gat = GeneralizationAcrossTime(clf=svc, predict_mode='mean-prediction')
    with warnings.catch_warnings(record=True):
        gat.fit(epochs)

    # sklearn needs it: c.f.
    # https://github.com/scikit-learn/scikit-learn/issues/2723
    # and http://bit.ly/1u7t8UT
    assert_raises(ValueError, gat.score, epochs2)
    gat.score(epochs)
    scores = sum(scores, [])  # flatten
    assert_true(0.0 <= np.min(scores) <= 1.0)
    assert_true(0.0 <= np.max(scores) <= 1.0)

    # Test that gets error if train on one dataset, test on another, and don't
    # specify appropriate cv:
    gat = GeneralizationAcrossTime()
    with warnings.catch_warnings(record=True):
        gat.fit(epochs)
    gat.predict(epochs)
    assert_raises(ValueError, gat.predict, epochs[:10])

    # Test combinations of complex scenarios
    # 2 or more distinct classes
    n_classes = [2, 4]  # 4 tested
    # nicely ordered labels or not
    le = LabelEncoder()
    y = le.fit_transform(epochs.events[:, 2])
    y[len(y) // 2:] += 2
    ys = (y, y + 1000)
    # Univariate and multivariate prediction
    svc = SVC(C=1, kernel='linear')

    class SVC_proba(SVC):
        def predict(self, x):
            probas = super(SVC_proba, self).predict_proba(x)
            return probas[:, 0]

    svcp = SVC_proba(C=1, kernel='linear', probability=True)
    clfs = [svc, svcp]
    scorers = [None, mean_squared_error]
    # Test all combinations
    for clf, scorer in zip(clfs, scorers):
        for y in ys:
            for n_class in n_classes:
                y_ = y % n_class
                with warnings.catch_warnings(record=True):
                    gat = GeneralizationAcrossTime(cv=2, clf=clf,
                                                   scorer=scorer)
                    gat.fit(epochs, y=y_)
                    gat.score(epochs, y=y_)
