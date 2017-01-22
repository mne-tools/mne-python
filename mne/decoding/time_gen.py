# Authors: Jean-Remi King <jeanremi.king@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Clement Moutard <clement.moutard@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import copy

from .base import _set_cv
from ..io.pick import _pick_data_channels
from ..viz.decoding import plot_gat_matrix, plot_gat_times
from ..parallel import parallel_func, check_n_jobs
from ..utils import warn, check_version


class _DecodingTime(dict):
    """Dictionary to configure the training times.

    It has the following keys:

    'slices' : ndarray, shape (n_clfs,)
        Array of time slices (in indices) used for each classifier.
        If not given, computed from 'start', 'stop', 'length', 'step'.
    'start' : float
        Time at which to start decoding (in seconds).
        Defaults to min(epochs.times).
    'stop' : float
        Maximal time at which to stop decoding (in seconds).
        Defaults to max(times).
    'step' : float
        Duration separating the start of subsequent classifiers (in
        seconds). Defaults to one time sample.
    'length' : float
        Duration of each classifier (in seconds). Defaults to one time sample.

    If None, empty dict.
    """

    def __repr__(self):  # noqa: D105
        s = ""
        if "start" in self:
            s += "start: %0.3f (s)" % (self["start"])
        if "stop" in self:
            s += ", stop: %0.3f (s)" % (self["stop"])
        if "step" in self:
            s += ", step: %0.3f (s)" % (self["step"])
        if "length" in self:
            s += ", length: %0.3f (s)" % (self["length"])
        if "slices" in self:
            # identify depth: training times only contains n_time but
            # testing_times can contain n_times or n_times * m_times
            depth = [len(ii) for ii in self["slices"]]
            if len(np.unique(depth)) == 1:  # if all slices have same depth
                if depth[0] == 1:  # if depth is one
                    s += ", n_time_windows: %s" % (len(depth))
                else:
                    s += ", n_time_windows: %s x %s" % (len(depth), depth[0])
            else:
                s += (", n_time_windows: %s x [%s, %s]" %
                      (len(depth),
                       min([len(ii) for ii in depth]),
                       max(([len(ii) for ii in depth]))))
        return "<DecodingTime | %s>" % s


class _GeneralizationAcrossTime(object):
    """Object to train and test classifiers at and acrosstime samples."""

    def __init__(self, picks=None, cv=5, clf=None, train_times=None,
                 test_times=None, predict_method='predict',
                 predict_mode='cross-validation', scorer=None,
                 score_mode='mean-fold-wise', n_jobs=1):  # noqa: D102

        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        # Store parameters in object
        self.cv = cv
        # Define training sliding window
        self.train_times = (_DecodingTime() if train_times is None
                            else _DecodingTime(train_times))
        # Define testing sliding window. If None, will be set in predict()
        if test_times is None:
            self.test_times = _DecodingTime()
        elif test_times == 'diagonal':
            self.test_times = 'diagonal'
        else:
            self.test_times = _DecodingTime(test_times)

        # Default classification pipeline
        if clf is None:
            scaler = StandardScaler()
            estimator = LogisticRegression()
            clf = Pipeline([('scaler', scaler), ('estimator', estimator)])
        self.clf = clf
        self.predict_mode = predict_mode
        self.scorer = scorer
        self.score_mode = score_mode
        self.picks = picks
        self.predict_method = predict_method
        self.n_jobs = n_jobs

    def fit(self, epochs, y=None):
        """Train a classifier on each specified time slice.

        .. note::
            This function sets the ``picks_``, ``ch_names``, ``cv_``,
            ``y_train``, ``train_times_`` and ``estimators_`` attributes.

        Parameters
        ----------
        epochs : instance of Epochs
            The epochs.
        y : list or ndarray of int, shape (n_samples,) or None, optional
            To-be-fitted model values. If None, y = epochs.events[:, 2].

        Returns
        -------
        self : GeneralizationAcrossTime
            Returns fitted GeneralizationAcrossTime object.

        Notes
        -----
        If X and y are not C-ordered and contiguous arrays of np.float64 and
        X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

        If X is a dense array, then the other methods will not support sparse
        matrices as input.
        """
        from sklearn.base import clone

        # Clean attributes
        for att in ['picks_', 'ch_names', 'y_train_', 'cv_', 'train_times_',
                    'estimators_', 'test_times_', 'y_pred_', 'y_true_',
                    'scores_', 'scorer_']:
            if hasattr(self, att):
                delattr(self, att)

        n_jobs = self.n_jobs
        # Extract data from MNE structure
        X, y, self.picks_ = _check_epochs_input(epochs, y, self.picks)
        self.ch_names = [epochs.ch_names[p] for p in self.picks_]

        # Prepare cross-validation
        self.cv_, self._cv_splits = _set_cv(self.cv, self.clf, X=X, y=y)

        self.y_train_ = y

        # Get train slices of times
        self.train_times_ = _sliding_window(epochs.times, self.train_times,
                                            epochs.info['sfreq'])

        # Parallel across training time
        # TODO: JRK: Chunking times points needs to be simplified
        parallel, p_func, n_jobs = parallel_func(_fit_slices, n_jobs)
        n_chunks = min(len(self.train_times_['slices']), n_jobs)
        time_chunks = np.array_split(self.train_times_['slices'], n_chunks)

        out = parallel(p_func(clone(self.clf),
                              X[..., np.unique(np.concatenate(time_chunk))],
                              y, time_chunk, self._cv_splits)
                       for time_chunk in time_chunks)

        # Unpack estimators into time slices X folds list of lists.
        self.estimators_ = sum(out, list())
        return self

    def predict(self, epochs):
        """Get predictions of classifiers on each specified testing time slice.

        .. note::
            This function sets the ``y_pred_`` and ``test_times_`` attributes.

        Parameters
        ----------
        epochs : instance of Epochs
            The epochs. Can be similar to fitted epochs or not. See
            predict_mode parameter.

        Returns
        -------
        y_pred : list of lists of arrays of floats, shape (n_train_t, n_test_t, n_epochs, n_prediction_dims)
            The single-trial predictions at each training time and each testing
            time. Note that the number of testing times per training time need
            not be regular; else
            ``np.shape(y_pred_) = (n_train_time, n_test_time, n_epochs)``.
        """  # noqa: E501
        # Check that classifier has predict_method (e.g. predict_proba is not
        # always available):
        if not hasattr(self.clf, self.predict_method):
            raise NotImplementedError('%s does not have "%s"' % (
                self.clf, self.predict_method))

        # Check that at least one classifier has been trained
        if not hasattr(self, 'estimators_'):
            raise RuntimeError('Please fit models before trying to predict')

        # Check predict mode
        if self.predict_mode not in ['cross-validation', 'mean-prediction']:
            raise ValueError('predict_mode must be a str, "mean-prediction" '
                             'or "cross-validation"')

        # Check that training cv and predicting cv match
        if self.predict_mode == 'cross-validation':
            n_est_cv = [len(estimator) for estimator in self.estimators_]
            heterogeneous_cv = len(set(n_est_cv)) != 1
            mismatch_cv = n_est_cv[0] != len(self._cv_splits)
            mismatch_y = len(self.y_train_) != len(epochs)
            if heterogeneous_cv or mismatch_cv or mismatch_y:
                raise ValueError(
                    'When predict_mode = "cross-validation", the training '
                    'and predicting cv schemes must be identical.')

            # Check that cv is a partition: i.e. that each tested sample may
            # have more than one prediction, such as with ShuffleSplit.
            test_idx = [ii for _, test in self._cv_splits for ii in test]
            if sum([sum(np.array(test_idx) == idx) > 1 for idx in test_idx]):
                raise ValueError('cv must be a partition if predict_mode is '
                                 '"cross-validation".')

        # Clean attributes
        for att in ['y_pred_', 'test_times_', 'scores_', 'scorer_', 'y_true_']:
            if hasattr(self, att):
                delattr(self, att)
        _warn_once.clear()  # reset self-baked warning tracker

        X, y, _ = _check_epochs_input(epochs, None, self.picks_)

        if not np.all([len(test) for train, test in self._cv_splits]):
            warn('Some folds do not have any test epochs.')

        # Define testing sliding window
        if self.test_times == 'diagonal':
            test_times = _DecodingTime()
            test_times['slices'] = [[s] for s in self.train_times_['slices']]
            test_times['times'] = [[s] for s in self.train_times_['times']]
        elif isinstance(self.test_times, dict):
            test_times = copy.deepcopy(self.test_times)
        else:
            raise ValueError('test_times must be a dict or "diagonal"')

        if 'slices' not in test_times:
            if 'length' not in self.train_times_.keys():
                ValueError('Need test_times["slices"] with adhoc train_times.')
            # Check that same number of time sample in testing than in training
            # (otherwise it won 't be the same number of features')
            test_times['length'] = test_times.get('length',
                                                  self.train_times_['length'])
            # Make a sliding window for each training time.
            slices_list = list()
            for _ in range(len(self.train_times_['slices'])):
                test_times_ = _sliding_window(epochs.times, test_times,
                                              epochs.info['sfreq'])
                slices_list += [test_times_['slices']]
            test_times = test_times_
            test_times['slices'] = slices_list
        test_times['times'] = [_set_window_time(test, epochs.times)
                               for test in test_times['slices']]

        for train, tests in zip(self.train_times_['slices'],
                                test_times['slices']):
            # The user may define irregular timing. We thus need to ensure
            # that the dimensionality of each estimator (i.e. training
            # time) corresponds to the dimensionality of each testing time)
            if not np.all([len(test) == len(train) for test in tests]):
                raise ValueError('train_times and test_times must '
                                 'have identical lengths')

        # Store all testing times parameters
        self.test_times_ = test_times

        n_orig_epochs, _, n_times = X.shape

        # Subselects the to-be-predicted epochs so as to manipulate a
        # contiguous array X by using slices rather than indices.
        test_epochs = []
        if self.predict_mode == 'cross-validation':
            test_idxs = [ii for train, test in self._cv_splits for ii in test]
            start = 0
            for _, test in self._cv_splits:
                n_test_epochs = len(test)
                stop = start + n_test_epochs
                test_epochs.append(slice(start, stop, 1))
                start += n_test_epochs
            X = X[test_idxs]

        # Prepare parallel predictions across testing time points
        # FIXME Note that this means that TimeDecoding.predict isn't parallel
        parallel, p_func, n_jobs = parallel_func(_predict_slices, self.n_jobs)
        n_test_slice = max(len(sl) for sl in self.test_times_['slices'])
        # Loop across estimators (i.e. training times)
        n_chunks = min(n_test_slice, n_jobs)
        chunks = [np.array_split(slices, n_chunks)
                  for slices in self.test_times_['slices']]
        chunks = map(list, zip(*chunks))

        # To minimize memory during parallelization, we apply some chunking
        y_pred = parallel(p_func(
            estimators=self.estimators_, cv_splits=self._cv_splits,
            predict_mode=self.predict_mode, predict_method=self.predict_method,
            n_orig_epochs=n_orig_epochs, test_epochs=test_epochs,
            **dict(zip(['X', 'train_times'], _chunk_data(X, chunk))))
            for chunk in chunks)

        # Concatenate chunks across test time dimension.
        n_tests = [len(sl) for sl in self.test_times_['slices']]
        if len(set(n_tests)) == 1:  # does GAT deal with a regular array/matrix
            self.y_pred_ = np.concatenate(y_pred, axis=1)
        else:
            # Non regular testing times, y_pred is an array of arrays with
            # different lengths.
            # FIXME: should do this with numpy operators only
            self.y_pred_ = [[test for chunk in train for test in chunk]
                            for train in map(list, zip(*y_pred))]
        return self.y_pred_

    def score(self, epochs=None, y=None):
        """Score Epochs.

        Estimate scores across trials by comparing the prediction estimated for
        each trial to its true value.

        Calls ``predict()`` if it has not been already.

        .. note::
            The function updates the ``scorer_``, ``scores_``, and
            ``y_true_`` attributes.

        .. note::
            If ``predict_mode`` is 'mean-prediction', ``score_mode`` is
            automatically set to 'mean-sample-wise'.

        Parameters
        ----------
        epochs : instance of Epochs | None, optional
            The epochs. Can be similar to fitted epochs or not.
            If None, it needs to rely on the predictions ``y_pred_``
            generated with ``predict()``.
        y : list | ndarray, shape (n_epochs,) | None, optional
            True values to be compared with the predictions ``y_pred_``
            generated with ``predict()`` via ``scorer_``.
            If None and ``predict_mode``=='cross-validation' y = ``y_train_``.

        Returns
        -------
        scores : list of lists of float
            The scores estimated by ``scorer_`` at each training time and each
            testing time (e.g. mean accuracy of ``predict(X)``). Note that the
            number of testing times per training time need not be regular;
            else, np.shape(scores) = (n_train_time, n_test_time). If
            ``score_mode`` is 'fold-wise', np.shape(scores) = (n_train_time,
            n_test_time, n_folds).
        """
        import sklearn.metrics
        from sklearn.base import is_classifier
        from sklearn.metrics import accuracy_score, mean_squared_error
        if check_version('sklearn', '0.17'):
            from sklearn.base import is_regressor
        else:
            def is_regressor(clf):
                return False

        # Run predictions if not already done
        if epochs is not None:
            self.predict(epochs)
        else:
            if not hasattr(self, 'y_pred_'):
                raise RuntimeError('Please predict() epochs first or pass '
                                   'epochs to score()')

        # Check scorer
        if self.score_mode not in ('fold-wise', 'mean-fold-wise',
                                   'mean-sample-wise'):
            raise ValueError("score_mode must be 'fold-wise', "
                             "'mean-fold-wise' or 'mean-sample-wise'. "
                             "Got %s instead'" % self.score_mode)
        score_mode = self.score_mode
        if (self.predict_mode == 'mean-prediction' and
                self.score_mode != 'mean-sample-wise'):
            warn("score_mode changed from %s set to 'mean-sample-wise' because"
                 " predict_mode is 'mean-prediction'." % self.score_mode)
            score_mode = 'mean-sample-wise'
        self.scorer_ = self.scorer
        if self.scorer_ is None:
            # Try to guess which scoring metrics should be used
            if self.predict_method == "predict":
                if is_classifier(self.clf):
                    self.scorer_ = accuracy_score
                elif is_regressor(self.clf):
                    self.scorer_ = mean_squared_error

        elif isinstance(self.scorer_, str):
            if hasattr(sklearn.metrics, '%s_score' % self.scorer_):
                self.scorer_ = getattr(sklearn.metrics, '%s_score' %
                                       self.scorer_)
            else:
                raise KeyError("{0} scorer Doesn't appear to be valid a "
                               "scikit-learn scorer.".format(self.scorer_))
        if not self.scorer_:
            raise ValueError('Could not find a scoring metric for clf=%s '
                             ' and predict_method=%s. Manually define scorer'
                             '.' % (self.clf, self.predict_method))

        # If no regressor is passed, use default epochs events
        if y is None:
            if self.predict_mode == 'cross-validation':
                y = self.y_train_
            else:
                if epochs is not None:
                    y = epochs.events[:, 2]
                else:
                    raise RuntimeError('y is undefined because '
                                       'predict_mode="mean-prediction" and '
                                       'epochs are missing. You need to '
                                       'explicitly specify y.')
            if not np.all(np.unique(y) == np.unique(self.y_train_)):
                raise ValueError('Classes (y) passed differ from classes used '
                                 'for training. Please explicitly pass your y '
                                 'for scoring.')
        elif isinstance(y, list):
            y = np.array(y)

        # Clean attributes
        for att in ['scores_', 'y_true_']:
            if hasattr(self, att):
                delattr(self, att)

        self.y_true_ = y  # to be compared with y_pred for scoring

        # Preprocessing for parallelization across training times; to avoid
        # overheads, we divide them in large chunks.
        n_jobs = min(len(self.y_pred_[0][0]), check_n_jobs(self.n_jobs))
        parallel, p_func, n_jobs = parallel_func(_score_slices, n_jobs)
        n_estimators = len(self.train_times_['slices'])
        n_chunks = min(n_estimators, n_jobs)
        chunks = np.array_split(range(len(self.train_times_['slices'])),
                                n_chunks)
        scores = parallel(p_func(
            self.y_true_, [self.y_pred_[train] for train in chunk],
            self.scorer_, score_mode, self._cv_splits)
            for chunk in chunks)
        # TODO: np.array scores from initialization JRK
        self.scores_ = np.array([score for chunk in scores for score in chunk])
        return self.scores_


_warn_once = dict()


def _predict_slices(X, train_times, estimators, cv_splits, predict_mode,
                    predict_method, n_orig_epochs, test_epochs):
    """Aux function of GeneralizationAcrossTime.

    Run classifiers predictions loop across time samples.

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_features, n_times)
        To-be-fitted data.
    estimators : list of array-like, shape (n_times, n_folds)
        List of array of scikit-learn classifiers fitted in cross-validation.
    cv_splits : list of tuples
        List of tuples of train and test array generated from cv.
    train_times : list
        List of list of slices selecting data from X from which is prediction
        is generated.
    predict_method : str
        Specifies prediction method for the estimator.
    predict_mode : {'cross-validation', 'mean-prediction'}
        Indicates how predictions are achieved with regards to the cross-
        validation procedure:
            'cross-validation' : estimates a single prediction per sample based
                on the unique independent classifier fitted in the cross-
                validation.
            'mean-prediction' : estimates k predictions per sample, based on
                each of the k-fold cross-validation classifiers, and average
                these predictions into a single estimate per sample.
        Default: 'cross-validation'
    n_orig_epochs : int
        Original number of predicted epochs before slice definition. Note
        that the number of epochs may have been cropped if the cross validation
        is not deterministic (e.g. with ShuffleSplit, we may only predict a
        subset of epochs).
    test_epochs : list of slices
        List of slices to select the tested epoched in the cv.
    """
    # Check inputs
    n_epochs, _, n_times = X.shape
    n_train = len(estimators)
    n_test = [len(test_t_idxs) for test_t_idxs in train_times]

    # Loop across training times (i.e. estimators)
    y_pred = None
    for train_t_idx, (estimator_cv, test_t_idxs) in enumerate(
            zip(estimators, train_times)):
        # Checks whether predict is based on contiguous windows of lengths = 1
        # time-sample, ranging across the entire times. In this case, we will
        # be able to vectorize the testing times samples.
        # Set expected start time if window length == 1
        start = np.arange(n_times)
        contiguous_start = np.array_equal([sl[0] for sl in test_t_idxs], start)
        window_lengths = np.unique([len(sl) for sl in test_t_idxs])
        vectorize_times = (window_lengths == 1) and contiguous_start
        if vectorize_times:
            # In vectorize mode, we avoid iterating over time test time indices
            test_t_idxs = [slice(start[0], start[-1] + 1, 1)]
        elif _warn_once.get('vectorization', True):
            # Only warn if multiple testing time
            if len(test_t_idxs) > 1:
                warn('Due to a time window with length > 1, unable to '
                     ' vectorize across testing times. This leads to slower '
                     'predictions compared to the length == 1 case.')
                _warn_once['vectorization'] = False

        # Iterate over testing times. If vectorize_times: 1 iteration.
        for ii, test_t_idx in enumerate(test_t_idxs):
            # Vectoring chan_times features in case of multiple time samples
            # given to the estimators.
            X_pred = X
            if not vectorize_times:
                X_pred = X[:, :, test_t_idx].reshape(n_epochs, -1)

            if predict_mode == 'mean-prediction':
                # Bagging: predict with each fold's estimator and combine
                # predictions.
                y_pred_ = _predict(X_pred, estimator_cv,
                                   vectorize_times=vectorize_times,
                                   predict_method=predict_method)
                # Initialize y_pred now we know its dimensionality
                if y_pred is None:
                    n_dim = y_pred_.shape[-1]
                    y_pred = _init_ypred(n_train, n_test, n_orig_epochs, n_dim)
                if vectorize_times:
                    # When vectorizing, we predict multiple time points at once
                    # to gain speed. The utput predictions thus correspond to
                    # different test time indices.
                    y_pred[train_t_idx][test_t_idx] = y_pred_
                else:
                    # Output predictions in a single test time column
                    y_pred[train_t_idx][ii] = y_pred_
            elif predict_mode == 'cross-validation':
                # Predict using the estimator corresponding to each fold
                for (_, test), test_epoch, estimator in zip(
                        cv_splits, test_epochs, estimator_cv):
                    if test.size == 0:  # see issue #2788
                        continue
                    y_pred_ = _predict(X_pred[test_epoch], [estimator],
                                       vectorize_times=vectorize_times,
                                       predict_method=predict_method)
                    # Initialize y_pred now we know its dimensionality
                    if y_pred is None:
                        n_dim = y_pred_.shape[-1]
                        y_pred = _init_ypred(n_train, n_test, n_orig_epochs,
                                             n_dim)
                    if vectorize_times:
                        # When vectorizing, we predict multiple time points at
                        # once to gain speed. The output predictions thus
                        # correspond to different test_t_idx columns.
                        y_pred[train_t_idx][test_t_idx, test, ...] = y_pred_
                    else:
                        # Output predictions in a single test_t_idx column
                        y_pred[train_t_idx][ii, test, ...] = y_pred_

    return y_pred


def _init_ypred(n_train, n_test, n_orig_epochs, n_dim):
    """Initialize the predictions for each train/test time points.

    Parameters
    ----------
    n_train : int
        Number of training time point (i.e. estimators)
    n_test : list of int
        List of number of testing time points for each estimator.
    n_orig_epochs : int
        Number of epochs passed to gat.predict()
    n_dim : int
        Number of dimensionality of y_pred. See np.shape(clf.predict(X))

    Returns
    -------
    y_pred : np.array, shape(n_train, n_test, n_orig_epochs, n_dim)
        Empty array.

    Notes
    -----
    The ``y_pred`` variable can only be initialized after the first
    prediction, because we can't know whether it is a a categorical output or a
    set of probabilistic estimates. If all train time points have the same
    number of testing time points, then y_pred is a matrix. Else it is an array
    of arrays.
    """
    if len(set(n_test)) == 1:
        y_pred = np.empty((n_train, n_test[0], n_orig_epochs, n_dim))
    else:
        y_pred = np.array([np.empty((this_n, n_orig_epochs, n_dim))
                           for this_n in n_test])
    return y_pred


def _score_slices(y_true, list_y_pred, scorer, score_mode, cv):
    """Loop across chunks of testing slices."""
    scores_list = list()
    for y_pred in list_y_pred:
        scores = list()
        for t, this_y_pred in enumerate(y_pred):
            if score_mode in ['mean-fold-wise', 'fold-wise']:
                # Estimate score within each fold
                scores_ = list()
                for train, test in cv:
                    scores_.append(scorer(y_true[test], this_y_pred[test]))
                scores_ = np.array(scores_)
                # Summarize score as average across folds
                if score_mode == 'mean-fold-wise':
                    scores_ = np.mean(scores_, axis=0)
            elif score_mode == 'mean-sample-wise':
                # Estimate score across all y_pred without cross-validation.
                scores_ = scorer(y_true, this_y_pred)
            scores.append(scores_)
        scores_list.append(scores)
    return scores_list


def _check_epochs_input(epochs, y, picks=None):
    """Aux function of GeneralizationAcrossTime.

    Format MNE data into scikit-learn X and y.

    Parameters
    ----------
    epochs : instance of Epochs
            The epochs.
    y : ndarray shape (n_epochs) | list shape (n_epochs) | None
        To-be-fitted model. If y is None, y == epochs.events.
    picks : array-like of int | None
        The channels indices to include. If None the data
        channels in info, except bad channels, are used.

    Returns
    -------
    X : ndarray, shape (n_epochs, n_selected_chans, n_times)
        To-be-fitted data.
    y : ndarray, shape (n_epochs,)
        To-be-fitted model.
    picks : array-like of int | None
        The channels indices to include. If None the data
        channels in info, except bad channels, are used.
    """
    if y is None:
        y = epochs.events[:, 2]
    elif isinstance(y, list):
        y = np.array(y)

    # Convert MNE data into trials x features x time matrix
    X = epochs.get_data()

    # Pick channels
    if picks is None:  # just use good data channels
        picks = _pick_data_channels(epochs.info, with_ref_meg=False)
    if isinstance(picks, (list, np.ndarray)):
        picks = np.array(picks, dtype=np.int)
    else:
        raise ValueError('picks must be a list or a numpy.ndarray of int')
    X = X[:, picks, :]

    # Check data sets
    assert X.shape[0] == y.shape[0]
    return X, y, picks


def _fit_slices(clf, x_chunk, y, slices, cv_splits):
    """Aux function of GeneralizationAcrossTime.

    Fit each classifier.

    Parameters
    ----------
    clf : scikit-learn classifier
        The classifier object.
    x_chunk : ndarray, shape (n_epochs, n_features, n_times)
        To-be-fitted data.
    y : list | array, shape (n_epochs,)
        To-be-fitted model.
    slices : list | array, shape (n_training_slice,)
        List of training slices, indicating time sample relative to X
    cv_splits : list of tuples
        List of (train, test) tuples generated from cv.split()

    Returns
    -------
    estimators : list of lists of estimators
        List of fitted scikit-learn classifiers corresponding to each training
        slice.
    """
    from sklearn.base import clone
    # Initialize
    n_epochs = len(x_chunk)
    estimators = list()
    # Identify the time samples of X_chunck corresponding to X
    values = np.unique([val for sl in slices for val in sl])
    # Loop across time slices
    for t_slice in slices:
        # Translate absolute time samples into time sample relative to x_chunk
        t_slice = np.array([np.where(ii == values)[0][0] for ii in t_slice])
        # Select slice
        X = x_chunk[..., t_slice]
        # Reshape data matrix to flatten features if multiple time samples.
        X = X.reshape(n_epochs, np.prod(X.shape[1:]))
        # Loop across folds
        estimators_ = list()
        for fold, (train, test) in enumerate(cv_splits):
            # Fit classifier
            clf_ = clone(clf)
            clf_.fit(X[train, :], y[train])
            estimators_.append(clf_)
        # Store classifier
        estimators.append(estimators_)
    return estimators


def _sliding_window(times, window, sfreq):
    """Aux function of GeneralizationAcrossTime.

    Define the slices on which to train each classifier. The user either define
    the time slices manually in window['slices'] or s/he passes optional params
    to set them from window['start'], window['stop'], window['step'] and
    window['length'].

    Parameters
    ----------
    times : ndarray, shape (n_times,)
        Array of times from MNE epochs.
    window : dict keys: ('start', 'stop', 'step', 'length')
        Either train or test times.

    Returns
    -------
    window : dict
        Dictionary to set training and testing times.

    See Also
    --------
    GeneralizationAcrossTime

    """
    import copy

    window = _DecodingTime(copy.deepcopy(window))

    # Default values
    time_slices = window.get('slices', None)
    # If the user hasn't manually defined the time slices, we'll define them
    # with ``start``, ``stop``, ``step`` and ``length`` parameters.
    if time_slices is None:
        window['start'] = window.get('start', times[0])
        window['stop'] = window.get('stop', times[-1])
        window['step'] = window.get('step', 1. / sfreq)
        window['length'] = window.get('length', 1. / sfreq)

        if not (times[0] <= window['start'] <= times[-1]):
            raise ValueError(
                'start (%.2f s) outside time range [%.2f, %.2f].' % (
                    window['start'], times[0], times[-1]))
        if not (times[0] <= window['stop'] <= times[-1]):
            raise ValueError(
                'stop (%.2f s) outside time range [%.2f, %.2f].' % (
                    window['stop'], times[0], times[-1]))
        if window['step'] < 1. / sfreq:
            raise ValueError('step must be >= 1 / sampling_frequency')
        if window['length'] < 1. / sfreq:
            raise ValueError('length must be >= 1 / sampling_frequency')
        if window['length'] > np.ptp(times):
            raise ValueError('length must be <= time range')

        # Convert seconds to index

        def find_t_idx(t):  # find closest time point
            return np.argmin(np.abs(np.asarray(times) - t))

        start = find_t_idx(window['start'])
        stop = find_t_idx(window['stop'])
        step = int(round(window['step'] * sfreq))
        length = int(round(window['length'] * sfreq))

        # For each training slice, give time samples to be included
        time_slices = [range(start, start + length)]
        while (time_slices[-1][0] + step) <= (stop - length + 1):
            start = time_slices[-1][0] + step
            time_slices.append(range(start, start + length))
        window['slices'] = time_slices
    window['times'] = _set_window_time(window['slices'], times)
    return window


def _set_window_time(slices, times):
    """Aux function to define time as the last training time point."""
    t_idx_ = [t[-1] for t in slices]
    return times[t_idx_]


def _predict(X, estimators, vectorize_times, predict_method):
    """Aux function of GeneralizationAcrossTime.

    Predict each classifier. If multiple classifiers are passed, average
    prediction across all classifiers to result in a single prediction per
    classifier.

    Parameters
    ----------
    estimators : ndarray, shape (n_folds,) | shape (1,)
        Array of scikit-learn classifiers to predict data.
    X : ndarray, shape (n_epochs, n_features, n_times)
        To-be-predicted data
    vectorize_times : bool
        If True, X can be vectorized to predict all times points at once
    predict_method : str
        Name of the method used to make predictions from the estimator. For
        example, both `predict_proba` and `predict` are supported for
        sklearn.linear_model.LogisticRegression. Note that the scorer must be
        adapted to the prediction outputs of the method. Defaults to 'predict'.

    Returns
    -------
    y_pred : ndarray, shape (n_epochs, m_prediction_dimensions)
        Classifier's prediction for each trial.
    """
    from scipy import stats
    from sklearn.base import is_classifier
    # Initialize results:

    orig_shape = X.shape
    n_epochs = orig_shape[0]
    n_times = orig_shape[-1]

    n_clf = len(estimators)

    # in simple case, we are predicting each time sample as if it
    # was a different epoch
    if vectorize_times:  # treat times as trials for optimization
        X = np.hstack(X).T  # XXX JRK: still 17% of cpu time
    n_epochs_tmp = len(X)

    # Compute prediction for each sub-estimator (i.e. per fold)
    # if independent, estimators = all folds
    for fold, clf in enumerate(estimators):
        _y_pred = getattr(clf, predict_method)(X)
        # See inconsistency in dimensionality: scikit-learn/scikit-learn#5058
        if _y_pred.ndim == 1:
            _y_pred = _y_pred[:, None]
        # initialize predict_results array
        if fold == 0:
            predict_size = _y_pred.shape[1]
            y_pred = np.ones((n_epochs_tmp, predict_size, n_clf))
        y_pred[:, :, fold] = _y_pred

    # Bagging: Collapse y_pred across folds if necessary (i.e. if independent)
    # XXX need API to identify how multiple predictions can be combined?
    if fold > 0:
        if is_classifier(clf) and (predict_method == 'predict'):
            y_pred, _ = stats.mode(y_pred, axis=2)
        else:
            y_pred = np.mean(y_pred, axis=2, keepdims=True)
    y_pred = y_pred[:, :, 0]
    # Format shape
    if vectorize_times:
        shape = [n_epochs, n_times, y_pred.shape[-1]]
        y_pred = y_pred.reshape(shape).transpose([1, 0, 2])
    return y_pred


class GeneralizationAcrossTime(_GeneralizationAcrossTime):
    """Generalize across time and conditions.

    Creates an estimator object used to 1) fit a series of classifiers on
    multidimensional time-resolved data, and 2) test the ability of each
    classifier to generalize across other time samples, as in [1]_.

    Parameters
    ----------
    picks : array-like of int | None
        The channels indices to include. If None the data
        channels in info, except bad channels, are used.
    cv : int | object
        If an integer is passed, it is the number of folds.
        Specific cross-validation objects can be passed, see
        scikit-learn.cross_validation module for the list of possible objects.
        If clf is a classifier, defaults to StratifiedKFold(n_folds=5), else
        defaults to KFold(n_folds=5).
    clf : object | None
        An estimator compliant with the scikit-learn API (fit & predict).
        If None the classifier will be a standard pipeline including
        StandardScaler and LogisticRegression with default parameters.
    train_times : dict | None
        A dictionary to configure the training times:

            * ``slices`` : ndarray, shape (n_clfs,)
                Array of time slices (in indices) used for each classifier.
                If not given, computed from 'start', 'stop', 'length', 'step'.
            * ``start`` : float
                Time at which to start decoding (in seconds).
                Defaults to min(epochs.times).
            * ``stop`` : float
                Maximal time at which to stop decoding (in seconds).
                Defaults to max(times).
            * ``step`` : float
                Duration separating the start of subsequent classifiers (in
                seconds). Defaults to one time sample.
            * ``length`` : float
                Duration of each classifier (in seconds).
                Defaults to one time sample.

        If None, empty dict.
    test_times : 'diagonal' | dict | None, optional
        Configures the testing times.
        If set to 'diagonal', predictions are made at the time at which
        each classifier is trained.
        If set to None, predictions are made at all time points.
        If set to dict, the dict should contain ``slices`` or be contructed in
        a similar way to train_times:

            ``slices`` : ndarray, shape (n_clfs,)
                Array of time slices (in indices) used for each classifier.
                If not given, computed from 'start', 'stop', 'length', 'step'.

        If None, empty dict.
    predict_method : str
        Name of the method used to make predictions from the estimator. For
        example, both `predict_proba` and `predict` are supported for
        sklearn.linear_model.LogisticRegression. Note that the scorer must be
        adapted to the prediction outputs of the method. Defaults to 'predict'.
    predict_mode : {'cross-validation', 'mean-prediction'}
        Indicates how predictions are achieved with regards to the cross-
        validation procedure:

            * ``cross-validation`` : estimates a single prediction per sample
                based on the unique independent classifier fitted in the
                cross-validation.

            * ``mean-prediction`` : estimates k predictions per sample, based
                on each of the k-fold cross-validation classifiers, and average
                these predictions into a single estimate per sample.

        Defaults to 'cross-validation'.
    scorer : object | None | str
        scikit-learn Scorer instance or str type indicating the name of the
        scorer such as ``accuracy``, ``roc_auc``. If None, set to ``accuracy``.
    score_mode : {'fold-wise', 'mean-fold-wise', 'mean-sample-wise'}
        Determines how the scorer is estimated:

            * ``fold-wise`` : returns the score obtained in each fold.

            * ``mean-fold-wise`` : returns the average of the fold-wise scores.

            * ``mean-sample-wise`` : returns score estimated across across all
                y_pred independently of the cross-validation. This method is
                faster than ``mean-fold-wise`` but less conventional, use at
                your own risk.

        Defaults to 'mean-fold-wise'.
    n_jobs : int
        Number of jobs to run in parallel. Defaults to 1.

    Attributes
    ----------
    ``picks_`` : array-like of int | None
        The channels indices to include.
    ch_names : list, array-like, shape (n_channels,)
        Names of the channels used for training.
    ``y_train_`` : list | ndarray, shape (n_samples,)
        The categories used for training.
    ``train_times_`` : dict
        A dictionary that configures the training times:

            * ``slices`` : ndarray, shape (n_clfs,)
                Array of time slices (in indices) used for each classifier.
                If not given, computed from 'start', 'stop', 'length', 'step'.

            * ``times`` : ndarray, shape (n_clfs,)
                The training times (in seconds).

    ``test_times_`` : dict
        A dictionary that configures the testing times for each training time:

            ``slices`` : ndarray, shape (n_clfs, n_testing_times)
                Array of time slices (in indices) used for each classifier.
            ``times`` : ndarray, shape (n_clfs, n_testing_times)
                The testing times (in seconds) for each training time.

    ``cv_`` : CrossValidation object
        The actual CrossValidation input depending on y.
    ``estimators_`` : list of list of scikit-learn.base.BaseEstimator subclasses.
        The estimators for each time point and each fold.
    ``y_pred_`` : list of lists of arrays of floats, shape (n_train_times, n_test_times, n_epochs, n_prediction_dims)
        The single-trial predictions estimated by self.predict() at each
        training time and each testing time. Note that the number of testing
        times per training time need not be regular, else
        ``np.shape(y_pred_) = (n_train_time, n_test_time, n_epochs).``
    ``y_true_`` : list | ndarray, shape (n_samples,)
        The categories used for scoring ``y_pred_``.
    ``scorer_`` : object
        scikit-learn Scorer instance.
    ``scores_`` : list of lists of float
        The scores estimated by ``self.scorer_`` at each training time and each
        testing time (e.g. mean accuracy of self.predict(X)). Note that the
        number of testing times per training time need not be regular;
        else, ``np.shape(scores) = (n_train_time, n_test_time)``.

    See Also
    --------
    TimeDecoding

    References
    ----------
    .. [1] Jean-Remi King, Alexandre Gramfort, Aaron Schurger, Lionel Naccache
       and Stanislas Dehaene, "Two distinct dynamic modes subtend the
       detection of unexpected sounds", PLoS ONE, 2014
       DOI: 10.1371/journal.pone.0085791

    .. versionadded:: 0.9.0
    """  # noqa: E501

    def __init__(self, picks=None, cv=5, clf=None, train_times=None,
                 test_times=None, predict_method='predict',
                 predict_mode='cross-validation', scorer=None,
                 score_mode='mean-fold-wise', n_jobs=1):  # noqa: D102
        super(GeneralizationAcrossTime, self).__init__(
            picks=picks, cv=cv, clf=clf, train_times=train_times,
            test_times=test_times, predict_method=predict_method,
            predict_mode=predict_mode, scorer=scorer, score_mode=score_mode,
            n_jobs=n_jobs)

    def __repr__(self):  # noqa: D105
        s = ''
        if hasattr(self, "estimators_"):
            s += "fitted, start : %0.3f (s), stop : %0.3f (s)" % (
                self.train_times_.get('start', np.nan),
                self.train_times_.get('stop', np.nan))
        else:
            s += 'no fit'
        if hasattr(self, 'y_pred_'):
            s += (", predicted %d epochs" % len(self.y_pred_[0][0]))
        else:
            s += ", no prediction"
        if hasattr(self, "estimators_") and hasattr(self, 'scores_'):
            s += ',\n '
        else:
            s += ', '
        if hasattr(self, 'scores_'):
            s += "scored"
            if callable(self.scorer_):
                s += " (%s)" % (self.scorer_.__name__)
        else:
            s += "no score"

        return "<GAT | %s>" % s

    def plot(self, title=None, vmin=None, vmax=None, tlim=None, ax=None,
             cmap='RdBu_r', show=True, colorbar=True,
             xlabel=True, ylabel=True):
        """Plot GeneralizationAcrossTime object.

        Plot the score of each classifier at each tested time window.

        Parameters
        ----------
        title : str | None
            Figure title.
        vmin : float | None
            Min color value for scores. If None, sets to min(``gat.scores_``).
        vmax : float | None
            Max color value for scores. If None, sets to max(``gat.scores_``).
        tlim : ndarray, (train_min, test_max) | None
            The time limits used for plotting.
        ax : object | None
            Plot pointer. If None, generate new figure.
        cmap : str | cmap object
            The color map to be used. Defaults to ``'RdBu_r'``.
        show : bool
            If True, the figure will be shown. Defaults to True.
        colorbar : bool
            If True, the colorbar of the figure is displayed. Defaults to True.
        xlabel : bool
            If True, the xlabel is displayed. Defaults to True.
        ylabel : bool
            If True, the ylabel is displayed. Defaults to True.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The figure.
        """
        return plot_gat_matrix(self, title=title, vmin=vmin, vmax=vmax,
                               tlim=tlim, ax=ax, cmap=cmap, show=show,
                               colorbar=colorbar, xlabel=xlabel, ylabel=ylabel)

    def plot_diagonal(self, title=None, xmin=None, xmax=None, ymin=None,
                      ymax=None, ax=None, show=True, color=None,
                      xlabel=True, ylabel=True, legend=True, chance=True,
                      label='Classif. score'):
        """Plot GeneralizationAcrossTime object.

        Plot each classifier score trained and tested at identical time
        windows.

        Parameters
        ----------
        title : str | None
            Figure title.
        xmin : float | None, optional
            Min time value.
        xmax : float | None, optional
            Max time value.
        ymin : float | None, optional
            Min score value. If None, sets to min(scores).
        ymax : float | None, optional
            Max score value. If None, sets to max(scores).
        ax : object | None
            Instance of mataplotlib.axes.Axis. If None, generate new figure.
        show : bool
            If True, the figure will be shown. Defaults to True.
        color : str
            Score line color.
        xlabel : bool
            If True, the xlabel is displayed. Defaults to True.
        ylabel : bool
            If True, the ylabel is displayed. Defaults to True.
        legend : bool
            If True, a legend is displayed. Defaults to True.
        chance : bool | float. Defaults to None
            Plot chance level. If True, chance level is estimated from the type
            of scorer.
        label : str
            Score label used in the legend. Defaults to 'Classif. score'.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The figure.
        """
        return plot_gat_times(self, train_time='diagonal', title=title,
                              xmin=xmin, xmax=xmax,
                              ymin=ymin, ymax=ymax, ax=ax, show=show,
                              color=color, xlabel=xlabel, ylabel=ylabel,
                              legend=legend, chance=chance, label=label)

    def plot_times(self, train_time, title=None, xmin=None, xmax=None,
                   ymin=None, ymax=None, ax=None, show=True, color=None,
                   xlabel=True, ylabel=True, legend=True, chance=True,
                   label='Classif. score'):
        """Plot GeneralizationAcrossTime object.

        Plot the scores of the classifier trained at specific training time(s).

        Parameters
        ----------
        train_time : float | list or array of float
            Plots scores of the classifier trained at train_time.
        title : str | None
            Figure title.
        xmin : float | None, optional
            Min time value.
        xmax : float | None, optional
            Max time value.
        ymin : float | None, optional
            Min score value. If None, sets to min(scores).
        ymax : float | None, optional
            Max score value. If None, sets to max(scores).
        ax : object | None
            Instance of mataplotlib.axes.Axis. If None, generate new figure.
        show : bool
            If True, the figure will be shown. Defaults to True.
        color : str or list of str
            Score line color(s).
        xlabel : bool
            If True, the xlabel is displayed. Defaults to True.
        ylabel : bool
            If True, the ylabel is displayed. Defaults to True.
        legend : bool
            If True, a legend is displayed. Defaults to True.
        chance : bool | float.
            Plot chance level. If True, chance level is estimated from the type
            of scorer.
        label : str
            Score label used in the legend. Defaults to 'Classif. score'.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The figure.
        """
        if not np.array(train_time).dtype is np.dtype('float'):
            raise ValueError('train_time must be float | list or array of '
                             'floats. Got %s.' % type(train_time))

        return plot_gat_times(self, train_time=train_time, title=title,
                              xmin=xmin, xmax=xmax,
                              ymin=ymin, ymax=ymax, ax=ax, show=show,
                              color=color, xlabel=xlabel, ylabel=ylabel,
                              legend=legend, chance=chance, label=label)


class TimeDecoding(_GeneralizationAcrossTime):
    """Train and test a series of classifiers at each time point.

    This will result in a score across time.

    Parameters
    ----------
    picks : array-like of int | None
        The channels indices to include. If None the data
        channels in info, except bad channels, are used.
    cv : int | object
        If an integer is passed, it is the number of folds.
        Specific cross-validation objects can be passed, see
        scikit-learn.cross_validation module for the list of possible objects.
        If clf is a classifier, defaults to StratifiedKFold(n_folds=5), else
        defaults to KFold(n_folds=5).
    clf : object | None
        An estimator compliant with the scikit-learn API (fit & predict).
        If None the classifier will be a standard pipeline including
        StandardScaler and a Logistic Regression with default parameters.
    times : dict | None
        A dictionary to configure the training times:

            ``slices`` : ndarray, shape (n_clfs,)
                Array of time slices (in indices) used for each classifier.
                If not given, computed from 'start', 'stop', 'length', 'step'.

            ``start`` : float
                Time at which to start decoding (in seconds). By default,
                min(epochs.times).

            ``stop`` : float
                Maximal time at which to stop decoding (in seconds). By
                default, max(times).

            ``step`` : float
                Duration separating the start of subsequent classifiers (in
                seconds). By default, equals one time sample.

            ``length`` : float
                Duration of each classifier (in seconds). By default, equals
                one time sample.

        If None, empty dict.
    predict_method : str
        Name of the method used to make predictions from the estimator. For
        example, both `predict_proba` and `predict` are supported for
        sklearn.linear_model.LogisticRegression. Note that the scorer must be
        adapted to the prediction outputs of the method. Defaults to 'predict'.
    predict_mode : {'cross-validation', 'mean-prediction'}
        Indicates how predictions are achieved with regards to the cross-
        validation procedure:

            * ``cross-validation`` : estimates a single prediction per sample
                based on the unique independent classifier fitted in the
                cross-validation.

            * ``mean-prediction`` : estimates k predictions per sample, based
                on each of the k-fold cross-validation classifiers, and average
                these predictions into a single estimate per sample.

        Defaults to 'cross-validation'.
    scorer : object | None | str
        scikit-learn Scorer instance or str type indicating the name of the
        scorer such as ``accuracy``, ``roc_auc``. If None, set to ``accuracy``.
    score_mode : {'fold-wise', 'mean-fold-wise', 'mean-sample-wise'}
        Determines how the scorer is estimated:

            * ``fold-wise`` : returns the score obtained in each fold.

            * ``mean-fold-wise`` : returns the average of the fold-wise scores.

            * ``mean-sample-wise`` : returns score estimated across across all
                y_pred independently of the cross-validation. This method is
                faster than ``mean-fold-wise`` but less conventional, use at
                your own risk.

        Defaults to 'mean-fold-wise'.
    n_jobs : int
        Number of jobs to run in parallel. Defaults to 1.

    Attributes
    ----------
    ``picks_`` : array-like of int | None
        The channels indices to include.
    ch_names : list, array-like, shape (n_channels,)
        Names of the channels used for training.
    ``y_train_`` : ndarray, shape (n_samples,)
        The categories used for training.
    ``times_`` : dict
        A dictionary that configures the training times:

            * ``slices`` : ndarray, shape (n_clfs,)
                Array of time slices (in indices) used for each classifier.
                If not given, computed from 'start', 'stop', 'length', 'step'.

            * ``times`` : ndarray, shape (n_clfs,)
                The training times (in seconds).

    ``cv_`` : CrossValidation object
        The actual CrossValidation input depending on y.
    ``estimators_`` : list of list of scikit-learn.base.BaseEstimator subclasses.
        The estimators for each time point and each fold.
    ``y_pred_`` : ndarray, shape (n_times, n_epochs, n_prediction_dims)
        Class labels for samples in X.
    ``y_true_`` : list | ndarray, shape (n_samples,)
        The categories used for scoring ``y_pred_``.
    ``scorer_`` : object
        scikit-learn Scorer instance.
    ``scores_`` : list of float, shape (n_times,)
        The scores (mean accuracy of self.predict(X) wrt. y.).

    See Also
    --------
    GeneralizationAcrossTime

    Notes
    -----
    The function is equivalent to the diagonal of GeneralizationAcrossTime()

    .. versionadded:: 0.10
    """  # noqa: E501

    def __init__(self, picks=None, cv=5, clf=None, times=None,
                 predict_method='predict', predict_mode='cross-validation',
                 scorer=None, score_mode='mean-fold-wise',
                 n_jobs=1):  # noqa: D102
        super(TimeDecoding, self).__init__(picks=picks, cv=cv, clf=clf,
                                           train_times=times,
                                           test_times='diagonal',
                                           predict_method=predict_method,
                                           predict_mode=predict_mode,
                                           scorer=scorer,
                                           score_mode=score_mode,
                                           n_jobs=n_jobs)
        self._clean_times()

    def __repr__(self):  # noqa: D105
        s = ''
        if hasattr(self, "estimators_"):
            s += "fitted, start : %0.3f (s), stop : %0.3f (s)" % (
                self.times_.get('start', np.nan),
                self.times_.get('stop', np.nan))
        else:
            s += 'no fit'
        if hasattr(self, 'y_pred_'):
            s += (", predicted %d epochs" % len(self.y_pred_[0]))
        else:
            s += ", no prediction"
        if hasattr(self, "estimators_") and hasattr(self, 'scores_'):
            s += ',\n '
        else:
            s += ', '
        if hasattr(self, 'scores_'):
            s += "scored"
            if callable(self.scorer_):
                s += " (%s)" % (self.scorer_.__name__)
        else:
            s += "no score"

        return "<TimeDecoding | %s>" % s

    def fit(self, epochs, y=None):
        """Train a classifier on each specified time slice.

        .. note::
            This function sets the ``picks_``, ``ch_names``, ``cv_``,
            ``y_train``, ``train_times_`` and ``estimators_`` attributes.

        Parameters
        ----------
        epochs : instance of Epochs
            The epochs.
        y : list or ndarray of int, shape (n_samples,) or None, optional
            To-be-fitted model values. If None, y = epochs.events[:, 2].

        Returns
        -------
        self : TimeDecoding
            Returns fitted TimeDecoding object.

        Notes
        -----
        If X and y are not C-ordered and contiguous arrays of np.float64 and
        X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

        If X is a dense array, then the other methods will not support sparse
        matrices as input.
        """
        self._prep_times()
        super(TimeDecoding, self).fit(epochs, y=y)
        self._clean_times()
        return self

    def predict(self, epochs):
        """Test each classifier on each specified testing time slice.

        .. note::
            This function sets the ``y_pred_`` and ``test_times_`` attributes.

        Parameters
        ----------
        epochs : instance of Epochs
            The epochs. Can be similar to fitted epochs or not. See
            predict_mode parameter.

        Returns
        -------
        y_pred : list of lists of arrays of floats, shape (n_times, n_epochs, n_prediction_dims)
            The single-trial predictions at each time sample.
        """  # noqa: E501
        self._prep_times()
        super(TimeDecoding, self).predict(epochs)
        self._clean_times()
        return self.y_pred_

    def score(self, epochs=None, y=None):
        """Score Epochs.

        Estimate scores across trials by comparing the prediction estimated for
        each trial to its true value.

        Calls ``predict()`` if it has not been already.

        .. note::
            The function updates the ``scorer_``, ``scores_``, and
            ``y_true_`` attributes.

        .. note::
            If ``predict_mode`` is 'mean-prediction', ``score_mode`` is
            automatically set to 'mean-sample-wise'.

        Parameters
        ----------
        epochs : instance of Epochs | None, optional
            The epochs. Can be similar to fitted epochs or not.
            If None, it needs to rely on the predictions ``y_pred_``
            generated with ``predict()``.
        y : list | ndarray, shape (n_epochs,) | None, optional
            True values to be compared with the predictions ``y_pred_``
            generated with ``predict()`` via ``scorer_``.
            If None and ``predict_mode``=='cross-validation' y = ``y_train_``.

        Returns
        -------
        scores : list of float, shape (n_times,)
            The scores estimated by ``scorer_`` at each time sample (e.g. mean
            accuracy of ``predict(X)``).
        """
        if epochs is not None:
            self.predict(epochs)
        else:
            if not hasattr(self, 'y_pred_'):
                raise RuntimeError('Please predict() epochs first or pass '
                                   'epochs to score()')
        self._prep_times()
        super(TimeDecoding, self).score(epochs=None, y=y)
        self._clean_times()
        return self.scores_

    def plot(self, title=None, xmin=None, xmax=None, ymin=None, ymax=None,
             ax=None, show=True, color=None, xlabel=True, ylabel=True,
             legend=True, chance=True, label='Classif. score'):
        """Plotting function.

        Predict each classifier. If multiple classifiers are passed, average
        prediction across all classifiers to result in a single prediction per
        classifier.

        Parameters
        ----------
        title : str | None
            Figure title.
        xmin : float | None, optional,
            Min time value.
        xmax : float | None, optional,
            Max time value.
        ymin : float
            Min score value. Defaults to 0.
        ymax : float
            Max score value. Defaults to 1.
        ax : object | None
            Instance of mataplotlib.axes.Axis. If None, generate new figure.
        show : bool
            If True, the figure will be shown. Defaults to True.
        color : str
            Score line color. Defaults to 'steelblue'.
        xlabel : bool
            If True, the xlabel is displayed. Defaults to True.
        ylabel : bool
            If True, the ylabel is displayed. Defaults to True.
        legend : bool
            If True, a legend is displayed. Defaults to True.
        chance : bool | float. Defaults to None
            Plot chance level. If True, chance level is estimated from the type
            of scorer.
        label : str
            Score label used in the legend. Defaults to 'Classif. score'.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The figure.
        """
        # XXX JRK: need cleanup in viz
        self._prep_times()
        fig = plot_gat_times(self, train_time='diagonal', title=title,
                             xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, ax=ax,
                             show=show, color=color, xlabel=xlabel,
                             ylabel=ylabel, legend=legend, chance=chance,
                             label=label)
        self._clean_times()
        return fig

    def _prep_times(self):
        """Auxiliary function to allow compatibility with GAT."""
        self.test_times = 'diagonal'
        if hasattr(self, 'times'):
            self.train_times = self.times
        if hasattr(self, 'times_'):
            self.train_times_ = self.times_
            self.test_times_ = _DecodingTime()
            self.test_times_['slices'] = [[slic] for slic in
                                          self.train_times_['slices']]
            self.test_times_['times'] = [[tim] for tim in
                                         self.train_times_['times']]
        if hasattr(self, 'scores_'):
            self.scores_ = [[score] for score in self.scores_]
        if hasattr(self, 'y_pred_'):
            self.y_pred_ = [[y_pred] for y_pred in self.y_pred_]

    def _clean_times(self):
        """Auxiliary function to allow compatibility with GAT."""
        if hasattr(self, 'train_times'):
            self.times = self.train_times
        if hasattr(self, 'train_times_'):
            self.times_ = self.train_times_
        for attr in ['test_times', 'train_times',
                     'test_times_', 'train_times_']:
            if hasattr(self, attr):
                delattr(self, attr)
        if hasattr(self, 'y_pred_'):
            self.y_pred_ = [y_pred[0] for y_pred in self.y_pred_]
        if hasattr(self, 'scores_'):
            self.scores_ = [score[0] for score in self.scores_]


def _chunk_data(X, slices):
    """Smart chunking to avoid memory overload.

    The parallelization is performed across time samples. To avoid overheads,
    the X data is splitted into large chunks of different time sizes. To
    avoid duplicating the memory load to each job, we only pass the time
    samples that are required by each job. The indices of the training times
    must be adjusted accordingly.
    """
    # from object array to list
    slices = [sl for sl in slices if len(sl)]
    selected_times = np.hstack([np.ravel(sl) for sl in slices])
    start = np.min(selected_times)
    stop = np.max(selected_times) + 1
    slices_chunk = [sl - start for sl in slices]
    X_chunk = X[:, :, start:stop]
    return X_chunk, slices_chunk
