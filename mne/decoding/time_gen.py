# Authors: Jean-Remi King <jeanremi.king@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Clement Moutard <clement.moutard@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import copy

from ..io.pick import pick_types
from ..viz.decoding import plot_gat_matrix, plot_gat_times
from ..parallel import parallel_func, check_n_jobs


class _DecodingTime(dict):
    """A dictionary to configure the training times that has the following keys:

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
    If None, empty dict. """

    def __repr__(self):
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
    """ see GeneralizationAcrossTime
    """  # noqa
    def __init__(self, picks=None, cv=5, clf=None, train_times=None,
                 test_times=None, predict_mode='cross-validation', scorer=None,
                 n_jobs=1):

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
        self.picks = picks
        self.n_jobs = n_jobs

    def fit(self, epochs, y=None):
        """ Train a classifier on each specified time slice.

        Note. This function sets the ``picks_``, ``ch_names``, ``cv_``,
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
        ------
        If X and y are not C-ordered and contiguous arrays of np.float64 and
        X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

        If X is a dense array, then the other methods will not support sparse
        matrices as input.
        """
        from sklearn.base import clone
        from sklearn.cross_validation import check_cv, StratifiedKFold

        # clean attributes
        for att in ['picks_', 'ch_names', 'y_train_', 'cv_', 'train_times_',
                    'estimators_', 'test_times_', 'y_pred_', 'y_true_',
                    'scores_', 'scorer_']:
            if hasattr(self, att):
                delattr(self, att)

        n_jobs = self.n_jobs
        # Extract data from MNE structure
        X, y, self.picks_ = _check_epochs_input(epochs, y, self.picks)
        self.ch_names = [epochs.ch_names[p] for p in self.picks_]

        cv = self.cv
        if isinstance(cv, (int, np.int)):
            cv = StratifiedKFold(y, cv)
        cv = check_cv(cv, X, y, classifier=True)
        self.cv_ = cv  # update CV

        self.y_train_ = y

        # Cross validation scheme
        # XXX Cross validation should later be transformed into a make_cv, and
        # defined in __init__
        self.train_times_ = copy.deepcopy(self.train_times)
        if 'slices' not in self.train_times_:
            self.train_times_ = _sliding_window(epochs.times, self.train_times)

        # Parallel across training time
        # TODO: JRK: Chunking times points needs to be simplified
        parallel, p_time_gen, n_jobs = parallel_func(_fit_slices, n_jobs)
        n_chunks = min(X.shape[2], n_jobs)
        splits = np.array_split(self.train_times_['slices'], n_chunks)

        def f(x):
            return np.unique(np.concatenate(x))

        out = parallel(p_time_gen(clone(self.clf),
                                  X[..., f(train_slices_chunk)],
                                  y, train_slices_chunk, cv)
                       for train_slices_chunk in splits)
        # Unpack estimators into time slices X folds list of lists.
        self.estimators_ = sum(out, list())
        return self

    def predict(self, epochs):
        """ Test each classifier on each specified testing time slice.

        .. note:: This function sets the ``y_pred_`` and ``test_times_``
                  attributes.

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
        """  # noqa

        # Check that at least one classifier has been trained
        if not hasattr(self, 'estimators_'):
            raise RuntimeError('Please fit models before trying to predict')

        # clean attributes
        for att in ['y_pred_', 'test_times_', 'scores_', 'scorer_', 'y_true_']:
            if hasattr(self, att):
                delattr(self, att)

        n_jobs = self.n_jobs

        X, y, _ = _check_epochs_input(epochs, None, self.picks_)

        # Define testing sliding window
        if self.test_times == 'diagonal':
            test_times = _DecodingTime()
            test_times['slices'] = [[s] for s in self.train_times_['slices']]
            test_times['times'] = [[s] for s in self.train_times_['times']]
        elif isinstance(self.test_times, dict):
            test_times = copy.deepcopy(self.test_times)
        else:
            raise ValueError('`test_times` must be a dict or "diagonal"')

        if 'slices' not in test_times:
            # Check that same number of time sample in testing than in training
            # (otherwise it won 't be the same number of features')
            if 'length' not in test_times:
                test_times['length'] = self.train_times_['length']
            if test_times['length'] != self.train_times_['length']:
                raise ValueError('`train_times` and `test_times` must have '
                                 'identical `length` keys')
            # Make a sliding window for each training time.
            slices_list = list()
            times_list = list()
            for t in range(0, len(self.train_times_['slices'])):
                test_times_ = _sliding_window(epochs.times, test_times)
                times_list += [test_times_['times']]
                slices_list += [test_times_['slices']]
            test_times = test_times_
            test_times['slices'] = slices_list
            test_times['times'] = times_list

        # Store all testing times parameters
        self.test_times_ = test_times

        # Prepare parallel predictions
        parallel, p_time_gen, n_jobs = parallel_func(_predict_slices, n_jobs)
        n_estimators = len(self.train_times_['slices'])
        # Loop across estimators (i.e. training times)
        n_chunks = min(n_estimators, n_jobs)
        splits = [np.array_split(slices, n_chunks)
                  for slices in self.test_times_['slices']]
        splits = map(list, zip(*splits))

        def chunk_X(X, slices):
            """Smart chunking to avoid memory overload"""
            slices = [sl for sl in slices]  # from object array to list
            start = np.min(slices)
            stop = np.max(slices) + 1
            slices_ = np.array(slices) - start
            X_ = X[:, :, start:stop]
            return (X_, self.estimators_, self.cv_, slices_.tolist(),
                    self.predict_mode)

        y_pred = parallel(p_time_gen(*chunk_X(X, slices))
                          for slices in splits)

        # concatenate chunks across test time dimension. Don't use
        # np.concatenate as this would need new memory allocations
        self.y_pred_ = [[test for chunk in train for test in chunk]
                        for train in map(list, zip(*y_pred))]
        return self.y_pred_

    def score(self, epochs=None, y=None):
        """Score Epochs

        Estimate scores across trials by comparing the prediction estimated for
        each trial to its true value.

        Calls ``predict()`` if it has not been already.

        Note. The function updates the ``scorer_``, ``scores_``, and
        ``y_true_`` attributes.

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
            else, np.shape(scores) = (n_train_time, n_test_time).
        """
        from sklearn.metrics import accuracy_score

        # Run predictions if not already done
        if epochs is not None:
            self.predict(epochs)
        else:
            if not hasattr(self, 'y_pred_'):
                raise RuntimeError('Please predict() epochs first or pass '
                                   'epochs to score()')

        # clean gat.score() attributes
        for att in ['scores_', 'scorer_', 'y_true_']:
            if hasattr(self, att):
                delattr(self, att)

        # Check scorer
        # XXX Need API to identify proper scorer from the clf
        self.scorer_ = accuracy_score if self.scorer is None else self.scorer

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
        self.y_true_ = y  # to be compared with y_pred for scoring

        # Preprocessing for parallelization
        n_jobs = min(len(self.y_pred_[0][0]), check_n_jobs(self.n_jobs))
        parallel, p_time_gen, n_jobs = parallel_func(_score_slices, n_jobs)
        n_estimators = len(self.train_times_['slices'])
        n_chunks = min(n_estimators, n_jobs)
        splits = np.array_split(range(len(self.train_times_['slices'])),
                                n_chunks)
        scores = parallel(
            p_time_gen(self.y_true_,
                       [self.y_pred_[train] for train in split],
                       self.scorer_)
            for split in splits)

        self.scores_ = [score for chunk in scores for score in chunk]
        return self.scores_


def _predict_slices(X, estimators, cv, slices, predict_mode):
    """Aux function of GeneralizationAcrossTime that loops across chunks of
    testing slices.
    """
    out = list()
    for this_estimator, this_slice in zip(estimators, slices):
        out.append(_predict_time_loop(X, this_estimator, cv, this_slice,
                                      predict_mode))
    return out


def _predict_time_loop(X, estimators, cv, slices, predict_mode):
    """Aux function of GeneralizationAcrossTime

    Run classifiers predictions loop across time samples.

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_features, n_times)
        To-be-fitted data.
    estimators : array-like, shape (n_times, n_folds)
        Array of scikit-learn classifiers fitted in cross-validation.
    slices : list
        List of slices selecting data from X from which is prediction is
        generated.
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
    """
    n_epochs = len(X)
    # Loop across testing slices
    y_pred = [list() for _ in range(len(slices))]

    # XXX EHN: This loop should be parallelized in a similar way to fit()
    for t, indices in enumerate(slices):
        # Flatten features in case of multiple time samples
        Xtrain = X[:, :, indices].reshape(
            n_epochs, np.prod(X[:, :, indices].shape[1:]))

        # Single trial predictions
        if predict_mode == 'cross-validation':
            # If predict within cross validation, only predict with
            # corresponding classifier, else predict with each fold's
            # classifier and average prediction.

            # Check that training cv and predicting cv match
            if (len(estimators) != len(cv)) or (cv.n != Xtrain.shape[0]):
                raise ValueError(
                    'When `predict_mode = "cross-validation"`, the training '
                    'and predicting cv schemes must be identical.')
            for k, (train, test) in enumerate(cv):
                # XXX I didn't manage to initialize correctly this array, as
                # its size depends on the the type of predictor and the
                # number of class.
                if k == 0:
                    y_pred_ = _predict(Xtrain[test, :], estimators[k:k + 1])
                    y_pred[t] = np.empty((n_epochs, y_pred_.shape[1]))
                    y_pred[t][test, :] = y_pred_
                y_pred[t][test, :] = _predict(Xtrain[test, :],
                                              estimators[k:k + 1])
        elif predict_mode == 'mean-prediction':
            y_pred[t] = _predict(Xtrain, estimators)
        else:
            raise ValueError('`predict_mode` must be a str, "mean-prediction"'
                             ' or "cross-validation"')
    return y_pred


def _score_slices(y_true, list_y_pred, scorer):
    """Aux function of GeneralizationAcrossTime that loops across chunks of
    testing slices.
    """
    scores_list = list()
    for y_pred in list_y_pred:
        scores = list()
        for t, this_y_pred in enumerate(y_pred):
            # Scores across trials
            scores.append(scorer(y_true, np.array(this_y_pred)))
        scores_list.append(scores)
    return scores_list


def _check_epochs_input(epochs, y, picks=None):
    """Aux function of GeneralizationAcrossTime

    Format MNE data into scikit-learn X and y

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
        picks = pick_types(epochs.info, meg=True, eeg=True, seeg=True,
                           eog=False, ecg=False, misc=False, stim=False,
                           ref_meg=False, exclude='bads')
    if isinstance(picks, (list, np.ndarray)):
        picks = np.array(picks, dtype=np.int)
    else:
        raise ValueError('picks must be a list or a numpy.ndarray of int')
    X = X[:, picks, :]

    # Check data sets
    assert X.shape[0] == y.shape[0]
    return X, y, picks


def _fit_slices(clf, x_chunk, y, slices, cv):
    """Aux function of GeneralizationAcrossTime

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
    cv : scikit-learn cross-validation generator
        A cross-validation generator to use.

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
    values = np.unique(np.concatenate(slices))
    indices = range(len(values))
    # Loop across time slices
    for t_slice in slices:
        # Translate absolute time samples into time sample relative to x_chunk
        for ii in indices:
            t_slice[t_slice == values[ii]] = indices[ii]
        # Select slice
        X = x_chunk[..., t_slice]
        # Reshape data matrix to flatten features in case of multiple time
        # samples.
        X = X.reshape(n_epochs, np.prod(X.shape[1:]))
        # Loop across folds
        estimators_ = list()
        for fold, (train, test) in enumerate(cv):
            # Fit classifier
            clf_ = clone(clf)
            clf_.fit(X[train, :], y[train])
            estimators_.append(clf_)
        # Store classifier
        estimators.append(estimators_)
    return estimators


def _sliding_window(times, window_params):
    """Aux function of GeneralizationAcrossTime

    Define the slices on which to train each classifier.

    Parameters
    ----------
    times : ndarray, shape (n_times,)
        Array of times from MNE epochs.
    window_params : dict keys: ('start', 'stop', 'step', 'length')
        Either train or test times. See GAT documentation.

    Returns
    -------
    time_pick : list
        List of training slices, indicating for each classifier the time
        sample (in indices of times) to be fitted on.
    """

    window_params = _DecodingTime(window_params)

    # Sampling frequency as int
    freq = (times[-1] - times[0]) / len(times)

    # Default values
    if ('slices' in window_params and
            all(k in window_params for k in
                ('start', 'stop', 'step', 'length'))):
        time_pick = window_params['slices']
    else:
        if 'start' not in window_params:
            window_params['start'] = times[0]
        if 'stop' not in window_params:
            window_params['stop'] = times[-1]
        if 'step' not in window_params:
            window_params['step'] = freq
        if 'length' not in window_params:
            window_params['length'] = freq

        if (window_params['start'] < times[0] or
                window_params['start'] > times[-1]):
            raise ValueError(
                '`start` (%.2f s) outside time range [%.2f, %.2f].' % (
                    window_params['start'], times[0], times[-1]))
        if (window_params['stop'] < times[0] or
                window_params['stop'] > times[-1]):
            raise ValueError(
                '`stop` (%.2f s) outside time range [%.2f, %.2f].' % (
                    window_params['stop'], times[0], times[-1]))
        if window_params['step'] < freq:
            raise ValueError('`step` must be >= 1 / sampling_frequency')
        if window_params['length'] < freq:
            raise ValueError('`length` must be >= 1 / sampling_frequency')
        if window_params['length'] > np.ptp(times):
            raise ValueError('`length` must be <= time range')

        # Convert seconds to index

        def find_time_idx(t):  # find closest time point
            return np.argmin(np.abs(np.asarray(times) - t))

        start = find_time_idx(window_params['start'])
        stop = find_time_idx(window_params['stop'])
        step = int(round(window_params['step'] / freq))
        length = int(round(window_params['length'] / freq))

        # For each training slice, give time samples to be included
        time_pick = [range(start, start + length)]
        while (time_pick[-1][0] + step) <= (stop - length + 1):
            start = time_pick[-1][0] + step
            time_pick.append(range(start, start + length))
        window_params['slices'] = time_pick

    # Keep last training times in milliseconds
    t_inds_ = [t[-1] for t in window_params['slices']]
    window_params['times'] = times[t_inds_]

    return window_params


def _predict(X, estimators):
    """Aux function of GeneralizationAcrossTime

    Predict each classifier. If multiple classifiers are passed, average
    prediction across all classifiers to result in a single prediction per
    classifier.

    Parameters
    ----------
    estimators : ndarray, shape (n_folds,) | shape (1,)
        Array of scikit-learn classifiers to predict data.
    X : ndarray, shape (n_epochs, n_features, n_times)
        To-be-predicted data
    Returns
    -------
    y_pred : ndarray, shape (n_epochs, m_prediction_dimensions)
        Classifier's prediction for each trial.
    """
    from scipy import stats
    from sklearn.base import is_classifier
    # Initialize results:
    n_epochs = X.shape[0]
    n_clf = len(estimators)

    # Compute prediction for each sub-estimator (i.e. per fold)
    # if independent, estimators = all folds
    for fold, clf in enumerate(estimators):
        _y_pred = clf.predict(X)
        # See inconsistency in dimensionality: scikit-learn/scikit-learn#5058
        if _y_pred.ndim == 1:
            _y_pred = _y_pred[:, None]
        # initialize predict_results array
        if fold == 0:
            predict_size = _y_pred.shape[1]
            y_pred = np.ones((n_epochs, predict_size, n_clf))
        y_pred[:, :, fold] = _y_pred

    # Collapse y_pred across folds if necessary (i.e. if independent)
    if fold > 0:
        # XXX need API to identify how multiple predictions can be combined?
        if is_classifier(clf):
            y_pred, _ = stats.mode(y_pred, axis=2)
        else:
            y_pred = np.mean(y_pred, axis=2)

    # Format shape
    y_pred = y_pred.reshape((n_epochs, predict_size))
    return y_pred


class GeneralizationAcrossTime(_GeneralizationAcrossTime):
    """Generalize across time and conditions

    Creates and estimator object used to 1) fit a series of classifiers on
    multidimensional time-resolved data, and 2) test the ability of each
    classifier to generalize across other time samples.

    Parameters
    ----------
    picks : array-like of int | None
        The channels indices to include. If None the data
        channels in info, except bad channels, are used.
    cv : int | object
        If an integer is passed, it is the number of folds.
        Specific cross-validation objects can be passed, see
        scikit-learn.cross_validation module for the list of possible objects.
        Defaults to 5.
    clf : object | None
        An estimator compliant with the scikit-learn API (fit & predict).
        If None the classifier will be a standard pipeline including
        StandardScaler and LogisticRegression with default parameters.
    train_times : dict | None
        A dictionary to configure the training times:

            ``slices`` : ndarray, shape (n_clfs,)
                Array of time slices (in indices) used for each classifier.
                If not given, computed from 'start', 'stop', 'length', 'step'.
            ``start`` : float
                Time at which to start decoding (in seconds).
                Defaults to min(epochs.times).
            ``stop`` : float
                Maximal time at which to stop decoding (in seconds).
                Defaults to max(times).
            ``step`` : float
                Duration separating the start of subsequent classifiers (in
                seconds). Defaults to one time sample.
            ``length`` : float
                Duration of each classifier (in seconds).
                Defaults to one time sample.

        If None, empty dict.
    test_times : 'diagonal' | dict | None, optional
        Configures the testing times.
        If set to 'diagonal', predictions are made at the time at which
        each classifier is trained.
        If set to None, predictions are made at all time points.
        If set to dict, the dict should contain ``slices`` or be contructed in
        a similar way to train_times::

            ``slices`` : ndarray, shape (n_clfs,)
                Array of time slices (in indices) used for each classifier.
                If not given, computed from 'start', 'stop', 'length', 'step'.

        If None, empty dict.
    predict_mode : {'cross-validation', 'mean-prediction'}
        Indicates how predictions are achieved with regards to the cross-
        validation procedure:

            ``cross-validation`` : estimates a single prediction per sample
                based on the unique independent classifier fitted in the
                cross-validation.
            ``mean-prediction`` : estimates k predictions per sample, based on
                each of the k-fold cross-validation classifiers, and average
                these predictions into a single estimate per sample.

        Default: 'cross-validation'
    scorer : object | None
        scikit-learn Scorer instance. If None, set to accuracy_score.
    n_jobs : int
        Number of jobs to run in parallel. Defaults to 1.

    Attributes
    ----------
    picks_ : array-like of int | None
        The channels indices to include.
    ch_names : list, array-like, shape (n_channels,)
        Names of the channels used for training.
    y_train_ : list | ndarray, shape (n_samples,)
        The categories used for training.
    train_times_ : dict
        A dictionary that configures the training times:

            ``slices`` : ndarray, shape (n_clfs,)
                Array of time slices (in indices) used for each classifier.
                If not given, computed from 'start', 'stop', 'length', 'step'.
            ``times`` : ndarray, shape (n_clfs,)
                The training times (in seconds).

    test_times_ : dict
        A dictionary that configures the testing times for each training time:

            ``slices`` : ndarray, shape (n_clfs, n_testing_times)
                Array of time slices (in indices) used for each classifier.
            ``times`` : ndarray, shape (n_clfs, n_testing_times)
                The testing times (in seconds) for each training time.

    cv_ : CrossValidation object
        The actual CrossValidation input depending on y.
    estimators_ : list of list of scikit-learn.base.BaseEstimator subclasses.
        The estimators for each time point and each fold.
    y_pred_ : list of lists of arrays of floats, shape (n_train_times, n_test_times, n_epochs, n_prediction_dims)
        The single-trial predictions estimated by self.predict() at each
        training time and each testing time. Note that the number of testing
        times per training time need not be regular, else
        ``np.shape(y_pred_) = (n_train_time, n_test_time, n_epochs).``
    y_true_ : list | ndarray, shape (n_samples,)
        The categories used for scoring ``y_pred_``.
    scorer_ : object
        scikit-learn Scorer instance.
    scores_ : list of lists of float
        The scores estimated by ``self.scorer_`` at each training time and each
        testing time (e.g. mean accuracy of self.predict(X)). Note that the
        number of testing times per training time need not be regular;
        else, ``np.shape(scores) = (n_train_time, n_test_time)``.

    See Also
    --------
    TimeDecoding

    Notes
    -----
    The function implements the method used in:

        Jean-Remi King, Alexandre Gramfort, Aaron Schurger, Lionel Naccache
        and Stanislas Dehaene, "Two distinct dynamic modes subtend the
        detection of unexpected sounds", PLoS ONE, 2014
        DOI: 10.1371/journal.pone.0085791

    .. versionadded:: 0.9.0
    """  # noqa
    def __init__(self, picks=None, cv=5, clf=None, train_times=None,
                 test_times=None, predict_mode='cross-validation', scorer=None,
                 n_jobs=1):
        super(GeneralizationAcrossTime, self).__init__(
            picks=picks, cv=cv, clf=clf, train_times=train_times,
            test_times=test_times, predict_mode=predict_mode, scorer=scorer,
            n_jobs=n_jobs)

    def __repr__(self):
        s = ''
        if hasattr(self, "estimators_"):
            s += "fitted, start : %0.3f (s), stop : %0.3f (s)" % (
                self.train_times_['start'], self.train_times_['stop'])
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
        """Plotting function of GeneralizationAcrossTime object

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
        """Plotting function of GeneralizationAcrossTime object

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
        """Plotting function of GeneralizationAcrossTime object

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
        if (not isinstance(train_time, float) and
            not (isinstance(train_time, (list, np.ndarray)) and
                 np.all([isinstance(time, float) for time in train_time]))):
            raise ValueError('train_time must be float | list or array of '
                             'floats. Got %s.' % type(train_time))

        return plot_gat_times(self, train_time=train_time, title=title,
                              xmin=xmin, xmax=xmax,
                              ymin=ymin, ymax=ymax, ax=ax, show=show,
                              color=color, xlabel=xlabel, ylabel=ylabel,
                              legend=legend, chance=chance, label=label)


class TimeDecoding(_GeneralizationAcrossTime):
    """Train and test a series of classifiers at each time point to obtain a
    score across time.

    Parameters
    ----------
    picks : array-like of int | None
        The channels indices to include. If None the data
        channels in info, except bad channels, are used.
    cv : int | object
        If an integer is passed, it is the number of folds.
        Specific cross-validation objects can be passed, see
        scikit-learn.cross_validation module for the list of possible objects.
        Defaults to 5.
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
    predict_mode : {'cross-validation', 'mean-prediction'}
        Indicates how predictions are achieved with regards to the cross-
        validation procedure:

            ``cross-validation`` : estimates a single prediction per sample
                based on the unique independent classifier fitted in the
                cross-validation.
            ``mean-prediction`` : estimates k predictions per sample, based on
                each of the k-fold cross-validation classifiers, and average
                these predictions into a single estimate per sample.

        Default: 'cross-validation'
    scorer : object | None
        scikit-learn Scorer instance. If None, set to accuracy_score.
    n_jobs : int
        Number of jobs to run in parallel. Defaults to 1.

    Attributes
    ----------
    picks_ : array-like of int | None
        The channels indices to include.
    ch_names : list, array-like, shape (n_channels,)
        Names of the channels used for training.
    y_train_ : ndarray, shape (n_samples,)
        The categories used for training.
    times_ : dict
        A dictionary that configures the training times:

            ``slices`` : ndarray, shape (n_clfs,)
                Array of time slices (in indices) used for each classifier.
                If not given, computed from 'start', 'stop', 'length', 'step'.
            ``times`` : ndarray, shape (n_clfs,)
                The training times (in seconds).

    cv_ : CrossValidation object
        The actual CrossValidation input depending on y.
    estimators_ : list of list of scikit-learn.base.BaseEstimator subclasses.
        The estimators for each time point and each fold.
    y_pred_ : ndarray, shape (n_times, n_epochs, n_prediction_dims)
        Class labels for samples in X.
    y_true_ : list | ndarray, shape (n_samples,)
        The categories used for scoring ``y_pred_``.
    scorer_ : object
        scikit-learn Scorer instance.
    scores_ : list of float, shape (n_times,)
        The scores (mean accuracy of self.predict(X) wrt. y.).

    See Also
    --------
    GeneralizationAcrossTime

    Notes
    -----
    The function is equivalent to the diagonal of GeneralizationAcrossTime()

    .. versionadded:: 0.10
    """

    def __init__(self, picks=None, cv=5, clf=None, times=None,
                 predict_mode='cross-validation', scorer=None, n_jobs=1):
        super(TimeDecoding, self).__init__(picks=picks, cv=cv, clf=None,
                                           train_times=times,
                                           test_times='diagonal',
                                           predict_mode=predict_mode,
                                           scorer=scorer, n_jobs=n_jobs)
        self._clean_times()

    def __repr__(self):
        s = ''
        if hasattr(self, "estimators_"):
            s += "fitted, start : %0.3f (s), stop : %0.3f (s)" % (
                self.times_['start'], self.times_['stop'])
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
        """ Train a classifier on each specified time slice.

        Note. This function sets the ``picks_``, ``ch_names``, ``cv_``,
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
        ------
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
        """ Test each classifier on each specified testing time slice.

        .. note:: This function sets the ``y_pred_`` and ``test_times_``
                  attributes.

        Parameters
        ----------
        epochs : instance of Epochs
            The epochs. Can be similar to fitted epochs or not. See
            predict_mode parameter.

        Returns
        -------
        y_pred : list of lists of arrays of floats, shape (n_times, n_epochs, n_prediction_dims)
            The single-trial predictions at each time sample.
        """  # noqa
        self._prep_times()
        super(TimeDecoding, self).predict(epochs)
        self._clean_times()
        return self.y_pred_

    def score(self, epochs=None, y=None):
        """Score Epochs

        Estimate scores across trials by comparing the prediction estimated for
        each trial to its true value.

        Calls ``predict()`` if it has not been already.

        Note. The function updates the ``scorer_``, ``scores_``, and
        ``y_true_`` attributes.

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
        """Plotting function

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
        """Auxiliary function to allow compability with GAT"""
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
        """Auxiliary function to allow compability with GAT"""
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
