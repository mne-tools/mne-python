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
from mne.utils import (requires_sklearn, requires_sklearn_0_15, slow_test,
                       run_tests_if_main, check_version)
from mne.decoding import GeneralizationAcrossTime, TimeDecoding


data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)
event_id_gen = dict(aud_l=2, vis_l=4)

warnings.simplefilter('always')


def make_epochs():
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
    return epochs


@slow_test
@requires_sklearn_0_15
def test_generalization_across_time():
    """Test time generalization decoding
    """
    from sklearn.svm import SVC
    # KernelRidge is used for testing 1) regression analyses 2) n-dimensional
    # predictions.
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import mean_squared_error, roc_auc_score
    if check_version('sklearn', '0.18'):
        from sklearn.model_selection import (KFold, StratifiedKFold,
                                             ShuffleSplit, LeaveOneLabelOut)
    else:
        from sklearn.cross_validation import (KFold, StratifiedKFold,
                                              ShuffleSplit, LeaveOneLabelOut)

    epochs = make_epochs()

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
    # test different predict function:
    gat = GeneralizationAcrossTime(predict_method='decision_function')
    gat.fit(epochs)
    # With classifier, the default cv is StratifiedKFold
    assert_true(gat.cv_.__class__ == StratifiedKFold)
    gat.predict(epochs)
    assert_array_equal(np.shape(gat.y_pred_), (15, 15, 14, 1))
    gat.predict_method = 'predict_proba'
    gat.predict(epochs)
    assert_array_equal(np.shape(gat.y_pred_), (15, 15, 14, 2))
    gat.predict_method = 'foo'
    assert_raises(NotImplementedError, gat.predict, epochs)
    gat.predict_method = 'predict'
    gat.predict(epochs)
    assert_array_equal(np.shape(gat.y_pred_), (15, 15, 14, 1))
    assert_equal("<GAT | fitted, start : -0.200 (s), stop : 0.499 (s), "
                 "predicted 14 epochs, no score>",
                 "%s" % gat)
    gat.score(epochs)
    assert_true(gat.scorer_.__name__ == 'accuracy_score')
    # check clf / predict_method combinations for which the scoring metrics
    # cannot be inferred.
    gat.scorer = None
    gat.predict_method = 'decision_function'
    assert_raises(ValueError, gat.score, epochs)
    # Check specifying y manually
    gat.predict_method = 'predict'
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
                 "0.050 (s), length: 0.050 (s), n_time_windows: 15>",
                 "%s" % gat.train_times_)
    assert_equal("<DecodingTime | start: -0.200 (s), stop: 0.499 (s), step: "
                 "0.050 (s), length: 0.050 (s), n_time_windows: 15 x 15>",
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
    with warnings.catch_warnings(record=True):  # not vectorizing
        scores = gat.score(epochs)
    assert_true(isinstance(scores, np.ndarray))  # type check
    assert_equal(len(scores[0]), len(scores))  # shape check
    assert_equal(len(gat.test_times_['slices'][0][0]), 2)
    # Decim training steps
    gat = GeneralizationAcrossTime(train_times={'step': .100})
    with warnings.catch_warnings(record=True):
        gat.fit(epochs)
    gat.score(epochs)
    assert_true(len(gat.scores_) == len(gat.estimators_) == 8)  # training time
    assert_equal(len(gat.scores_[0]), 15)  # testing time

    # Test start stop training & test cv without n_fold params
    y_4classes = np.hstack((epochs.events[:7, 2], epochs.events[7:, 2] + 1))
    train_times = dict(start=0.090, stop=0.250)
    if check_version('sklearn', '0.18'):
        cv = LeaveOneLabelOut()
        # XXX we cannot pass any other parameters than X and y to cv.split
        # so we have to build it before hand
        cv = [(train, test) for train, test in cv.split(
            X=y_4classes, y=y_4classes, labels=y_4classes)]
    else:
        cv = LeaveOneLabelOut(y_4classes)
    gat = GeneralizationAcrossTime(cv=cv, train_times=train_times)
    # predict without fit
    assert_raises(RuntimeError, gat.predict, epochs)
    with warnings.catch_warnings(record=True):
        gat.fit(epochs, y=y_4classes)
    gat.score(epochs)
    assert_equal(len(gat.scores_), 4)
    assert_equal(gat.train_times_['times'][0], epochs.times[6])
    assert_equal(gat.train_times_['times'][-1], epochs.times[9])

    # Test score without passing epochs & Test diagonal decoding
    gat = GeneralizationAcrossTime(test_times='diagonal')
    with warnings.catch_warnings(record=True):  # not vectorizing
        gat.fit(epochs)
    assert_raises(RuntimeError, gat.score)
    with warnings.catch_warnings(record=True):  # not vectorizing
        gat.predict(epochs)
    scores = gat.score()
    assert_true(scores is gat.scores_)
    assert_equal(np.shape(gat.scores_), (15, 1))
    assert_array_equal([tim for ttime in gat.test_times_['times']
                        for tim in ttime], gat.train_times_['times'])
    from mne.utils import set_log_level
    set_log_level('error')
    # Test generalization across conditions
    gat = GeneralizationAcrossTime(predict_mode='mean-prediction', cv=2)
    with warnings.catch_warnings(record=True):
        gat.fit(epochs[0:6])
    with warnings.catch_warnings(record=True):
        # There are some empty test folds because of n_trials
        gat.predict(epochs[7:])
        gat.score(epochs[7:])

    # Test training time parameters
    gat_ = copy.deepcopy(gat)
    # --- start stop outside time range
    gat_.train_times = dict(start=-999.)
    assert_raises(ValueError, gat_.fit, epochs)
    gat_.train_times = dict(start=999.)
    assert_raises(ValueError, gat_.fit, epochs)
    # --- impossible slices
    gat_.train_times = dict(step=.000001)
    assert_raises(ValueError, gat_.fit, epochs)
    gat_.train_times = dict(length=.000001)
    assert_raises(ValueError, gat_.fit, epochs)
    gat_.train_times = dict(length=999.)
    assert_raises(ValueError, gat_.fit, epochs)

    # Test testing time parameters
    # --- outside time range
    gat.test_times = dict(start=-999.)
    with warnings.catch_warnings(record=True):  # no epochs in fold
        assert_raises(ValueError, gat.predict, epochs)
    gat.test_times = dict(start=999.)
    with warnings.catch_warnings(record=True):  # no test epochs
        assert_raises(ValueError, gat.predict, epochs)
    # --- impossible slices
    gat.test_times = dict(step=.000001)
    with warnings.catch_warnings(record=True):  # no test epochs
        assert_raises(ValueError, gat.predict, epochs)
    gat_ = copy.deepcopy(gat)
    gat_.train_times_['length'] = .000001
    gat_.test_times = dict(length=.000001)
    with warnings.catch_warnings(record=True):  # no test epochs
        assert_raises(ValueError, gat_.predict, epochs)
    # --- test time region of interest
    gat.test_times = dict(step=.150)
    with warnings.catch_warnings(record=True):  # not vectorizing
        gat.predict(epochs)
    assert_array_equal(np.shape(gat.y_pred_), (15, 5, 14, 1))
    # --- silly value
    gat.test_times = 'foo'
    with warnings.catch_warnings(record=True):  # no test epochs
        assert_raises(ValueError, gat.predict, epochs)
    assert_raises(RuntimeError, gat.score)
    # --- unmatched length between training and testing time
    gat.test_times = dict(length=.150)
    assert_raises(ValueError, gat.predict, epochs)
    # --- irregular length training and testing times
    train_times = dict(slices=[[0, 1], [1]])
    test_times = dict(slices=[[[0, 1]], [[0], [1]]])
    gat = GeneralizationAcrossTime(train_times=train_times,
                                   test_times=test_times)
    gat.fit(epochs)
    with warnings.catch_warnings(record=True):  # not vectorizing
        gat.score(epochs)
    assert_array_equal(np.shape(gat.y_pred_[0]), [1, len(epochs), 1])
    assert_array_equal(np.shape(gat.y_pred_[1]), [2, len(epochs), 1])
    # check cannot Automatically infer testing times for adhoc training times
    gat.test_times = None
    assert_raises(ValueError, gat.predict, epochs)

    svc = SVC(C=1, kernel='linear', probability=True)
    gat = GeneralizationAcrossTime(clf=svc, predict_mode='mean-prediction')
    with warnings.catch_warnings(record=True):
        gat.fit(epochs)

    # sklearn needs it: c.f.
    # https://github.com/scikit-learn/scikit-learn/issues/2723
    # and http://bit.ly/1u7t8UT
    assert_raises(ValueError, gat.score, epochs2)
    gat.score(epochs)
    assert_true(0.0 <= np.min(scores) <= 1.0)
    assert_true(0.0 <= np.max(scores) <= 1.0)

    # Test that gets error if train on one dataset, test on another, and don't
    # specify appropriate cv:
    if check_version('sklearn', '0.18'):
        cv = ShuffleSplit()
    else:  # XXX sklearn < 0.18
        cv = ShuffleSplit(len(epochs))
    gat = GeneralizationAcrossTime(cv=cv)
    gat.fit(epochs)
    with warnings.catch_warnings(record=True):
        gat.fit(epochs)

    gat.predict(epochs)
    assert_raises(ValueError, gat.predict, epochs[:10])

    # Make CV with some empty train and test folds:
    # --- empty test fold(s) should warn when gat.predict()
    gat._cv_splits[0] = [gat._cv_splits[0][0], np.empty(0)]
    with warnings.catch_warnings(record=True) as w:
        gat.predict(epochs)
        assert_true(len(w) > 0)
        assert_true(any('do not have any test epochs' in str(ww.message)
                        for ww in w))
    # --- empty train fold(s) should raise when gat.fit()
    gat = GeneralizationAcrossTime(cv=[([0], [1]), ([], [0])])
    assert_raises(ValueError, gat.fit, epochs[:2])

    # Check that still works with classifier that output y_pred with
    # shape = (n_trials, 1) instead of (n_trials,)
    if check_version('sklearn', '0.17'):  # no is_regressor before v0.17
        gat = GeneralizationAcrossTime(clf=KernelRidge(), cv=2)
        epochs.crop(None, epochs.times[2])
        gat.fit(epochs)
        # With regression the default cv is KFold and not StratifiedKFold
        assert_true(gat.cv_.__class__ == KFold)
        gat.score(epochs)
        # with regression the default scoring metrics is mean squared error
        assert_true(gat.scorer_.__name__ == 'mean_squared_error')

    # Test combinations of complex scenarios
    # 2 or more distinct classes
    n_classes = [2, 4]  # 4 tested
    # nicely ordered labels or not
    le = LabelEncoder()
    y = le.fit_transform(epochs.events[:, 2])
    y[len(y) // 2:] += 2
    ys = (y, y + 1000)
    # Univariate and multivariate prediction
    svc = SVC(C=1, kernel='linear', probability=True)
    reg = KernelRidge()

    def scorer_proba(y_true, y_pred):
        roc_auc_score(y_true, y_pred[:, 0])

    # We re testing 3 scenario: default, classifier + predict_proba, regressor
    scorers = [None, scorer_proba, None]
    predict_methods = [None, 'predict_proba', None]
    clfs = [svc, svc, reg]
    # Test all combinations
    for clf, predict_method, scorer in zip(clfs, predict_methods, scorers):
        for y in ys:
            for n_class in n_classes:
                for predict_mode in ['cross-validation', 'mean-prediction']:
                    y_ = y % n_class
                    with warnings.catch_warnings(record=True):
                        gat = GeneralizationAcrossTime(
                            cv=2, clf=clf, scorer=scorer,
                            predict_mode=predict_mode)
                        gat.fit(epochs, y=y_)
                        gat.score(epochs, y=y_)


@requires_sklearn
def test_decoding_time():
    """Test TimeDecoding
    """
    from sklearn.svm import SVR
    from sklearn.cross_validation import KFold
    epochs = make_epochs()
    tg = TimeDecoding()
    assert_equal("<TimeDecoding | no fit, no prediction, no score>", '%s' % tg)
    assert_true(hasattr(tg, 'times'))
    assert_true(not hasattr(tg, 'train_times'))
    assert_true(not hasattr(tg, 'test_times'))
    tg.fit(epochs)
    assert_equal("<TimeDecoding | fitted, start : -0.200 (s), stop : 0.499 "
                 "(s), no prediction, no score>", '%s' % tg)
    assert_true(not hasattr(tg, 'train_times_'))
    assert_true(not hasattr(tg, 'test_times_'))
    assert_raises(RuntimeError, tg.score, epochs=None)
    with warnings.catch_warnings(record=True):  # not vectorizing
        tg.predict(epochs)
    assert_equal("<TimeDecoding | fitted, start : -0.200 (s), stop : 0.499 "
                 "(s), predicted 14 epochs, no score>",
                 '%s' % tg)
    assert_array_equal(np.shape(tg.y_pred_), [15, 14, 1])
    with warnings.catch_warnings(record=True):  # not vectorizing
        tg.score(epochs)
    tg.score()
    assert_array_equal(np.shape(tg.scores_), [15])
    assert_equal("<TimeDecoding | fitted, start : -0.200 (s), stop : 0.499 "
                 "(s), predicted 14 epochs,\n scored (accuracy_score)>",
                 '%s' % tg)
    # Test with regressor
    clf = SVR()
    cv = KFold(len(epochs))
    y = np.random.rand(len(epochs))
    tg = TimeDecoding(clf=clf, cv=cv)
    tg.fit(epochs, y=y)

run_tests_if_main()
