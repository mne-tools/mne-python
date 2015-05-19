# Authors: Jean-Remi King <jeanremi.king@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Clement Moutard <clement.moutard@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from ..io.pick import pick_types
from ..viz.decoding import plot_gat_matrix, plot_gat_times
from ..parallel import parallel_func, check_n_jobs
from ..utils import logger, verbose, deprecated


class _DecodingTime(dict):
    """A dictionary to configure the training times that has the following keys:

    'slices' : np.ndarray, shape (n_clfs,)
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
    If None, empty dict. Defaults to None."""

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


class GeneralizationAcrossTime(object):
    """Generalize across time and conditions

    Creates and estimator object used to 1) fit a series of classifiers on
    multidimensional time-resolved data, and 2) test the ability of each
    classifier to generalize across other time samples.

    Parameters
    ----------
    cv : int | object
        If an integer is passed, it is the number of folds.
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects.
        Defaults to 5.
    clf : object | None
        An estimator compliant with the scikit-learn API (fit & predict).
        If None the classifier will be a standard pipeline including
        StandardScaler and a linear SVM with default parameters.
    train_times : dict | None
        A dictionary to configure the training times:

            ``slices`` : np.ndarray, shape (n_clfs,)
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

        If None, empty dict. Defaults to None.
    test_times : 'diagonal' | dict | None, optional
        Configures the testing times.
        If set to 'diagonal', predictions are made at the time at which
        each classifier is trained.
        If set to None, predictions are made at all time points.
        If set to dict, the dict should contain ``slices`` or be contructed in
        a similar way to train_times
            ``slices`` : np.ndarray, shape (n_clfs,)
                Array of time slices (in indices) used for each classifier.
                If not given, computed from 'start', 'stop', 'length', 'step'.

        If None, empty dict. Defaults to None.
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
    scorer : object
        scikit-learn Scorer instance. Default: accuracy_score.
    n_jobs : int
        Number of jobs to run in parallel. Defaults to 1.

    Attributes
    ----------
    y_train_ : list | np.ndarray, shape (n_samples,)
        The categories used for training.
    estimators_ : list of list of sklearn.base.BaseEstimator subclasses.
        The estimators for each time point and each fold.
    y_pred_ : list of lists of arrays of floats,
              shape (n_train_times, n_test_times, n_epochs, n_prediction_dims)
        The single-trial predictions estimated by self.predict() at each
        training time and each testing time. Note that the number of testing
        times per training time need not be regular, else
        np.shape(y_pred_) = [n_train_time, n_test_time, n_epochs].
    scores_ : list of lists of float
        The scores estimated by self.scorer at each training time and each
        testing time (e.g. mean accuracy of self.predict(X)). Note that the
        number of testing times per training time need not be regular;
        else, np.shape(scores) = [n_train_time, n_test_time].
    cv_ : CrossValidation object
        The actual CrossValidation input depending on y.
    picks_ : np.array, shape (n_channels,)
        Indices of the channels used for training.
    ch_names : list, shape (n_channels,)
        Names of the channels used for training.

    Notes
    -----
    The function implements the method used in:

        Jean-Remi King, Alexandre Gramfort, Aaron Schurger, Lionel Naccache
        and Stanislas Dehaene, "Two distinct dynamic modes subtend the
        detection of unexpected sounds", PLOS ONE, 2013

    .. versionadded:: 0.9.0
    """  # noqa
    def __init__(self, cv=5, clf=None, train_times=None, test_times=None,
                 predict_mode='cross-validation', scorer=None, n_jobs=1):

        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        # Store parameters in object
        self.cv = cv
        # Define training sliding window
        self.train_times = (_DecodingTime() if train_times is None
                            else _DecodingTime(train_times))
        # Define testing sliding window. If None, will be set in predict()
        self.test_times = test_times

        # Default classification pipeline
        if clf is None:
            scaler = StandardScaler()
            estimator = LogisticRegression()
            clf = Pipeline([('scaler', scaler), ('estimator', estimator)])
        self.clf = clf
        self.predict_mode = predict_mode
        self.scorer = scorer
        self.n_jobs = n_jobs

    def __repr__(self):
        s = ''
        if hasattr(self, "estimators_"):
            s += "fitted, start : %0.3f (s), stop : %0.3f (s)" % (
                self.train_times['start'], self.train_times['stop'])
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
            if callable(self.scorer):
                s += " (%s)" % (self.scorer.__name__)
        else:
            s += "no score"

        return "<GAT | %s>" % s

    def fit(self, epochs, y=None, picks=None):
        """ Train a classifier on each specified time slice.

        Parameters
        ----------
        epochs : instance of Epochs
            The epochs.
        y : list or np.ndarray of int, shape (n_samples,) or None, optional
            To-be-fitted model values. If None, y = epochs.events[:, 2].
            Defaults to None.
        picks : array-like of int | None, optional
            Channels to be included. If None only good data channels are used.
            Defaults to None.

        Returns
        -------
        self : object
            Returns self.

        Notes
        ------
        If X and y are not C-ordered and contiguous arrays of np.float64 and
        X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

        If X is a dense array, then the other methods will not support sparse
        matrices as input.
        """
        from sklearn.base import clone
        from sklearn.cross_validation import check_cv, StratifiedKFold

        # clean in case gat.fit() is called at unexpected moments
        for att in ['y_pred_', 'scores_', 'scorer_', 'y_true_']:
            if hasattr(self, att):
                delattr(self, att)

        n_jobs = self.n_jobs
        # Extract data from MNE structure
        X, y, picks = _check_epochs_input(epochs, y, picks)
        self.picks_ = picks
        self.ch_names = [epochs.ch_names[p] for p in picks]

        cv = self.cv
        if isinstance(cv, (int, np.int)):
            cv = StratifiedKFold(y, cv)
        cv = check_cv(cv, X, y, classifier=True)
        self.cv_ = cv  # update CV

        self.y_train_ = y

        # Cross validation scheme
        # XXX Cross validation should later be transformed into a make_cv, and
        # defined in __init__
        if 'slices' not in self.train_times:
            self.train_times['slices'] = _sliding_window(
                epochs.times, self.train_times)

        # Keep last training times in milliseconds
        t_inds_ = [t[-1] for t in self.train_times['slices']]
        self.train_times['times_'] = epochs.times[t_inds_]

        # Parallel across training time
        parallel, p_time_gen, n_jobs = parallel_func(_fit_slices, n_jobs)
        n_chunks = min(X.shape[2], n_jobs)
        splits = np.array_split(self.train_times['slices'], n_chunks)

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

        Note. This function sets and updates the ``y_pred_`` and the
        ``test_times`` attribute.

        Parameters
        ----------
        epochs : instance of Epochs
            The epochs. Can be similar to fitted epochs or not. See
            predict_mode parameter.

        Returns
        -------
        y_pred : list of lists of arrays of floats,
                 shape (n_train_t, n_test_t, n_epochs, n_prediction_dims)
            The single-trial predictions at each training time and each testing
            time. Note that the number of testing times per training time need
            not be regular;
            else, np.shape(y_pred_) = [n_train_time, n_test_time, n_epochs].
        """

        # clean in case gat.predict() is called at unexpected moments
        for att in ['scores_', 'scorer_', 'y_true_']:
            if hasattr(self, att):
                delattr(self, att)

        n_jobs = self.n_jobs

        # Check that at least one classifier has been trained
        if not hasattr(self, 'estimators_'):
            raise RuntimeError('Please fit models before trying to predict')

        cv = self.cv_  # Retrieve CV scheme from fit()
        X, y, _ = _check_epochs_input(epochs, None, self.picks_)

        # Define testing sliding window
        if self.test_times == 'diagonal':
            test_times_ = _DecodingTime()
            test_times_['slices'] = [[s] for s in self.train_times['slices']]
        elif self.test_times is None:
            test_times_ = _DecodingTime()
        elif isinstance(self.test_times, dict):
            test_times_ = self.test_times
        else:
            raise ValueError('`test_times` must be a dict, "diagonal" or None')

        if 'slices' not in test_times_:
            # Initialize array
            test_times_['slices_'] = list()
            # Force same number of time sample in testing than in training
            # (otherwise it won 't be the same number of features')
            test_times_['length'] = self.train_times['length']
            # Make a sliding window for each training time.
            for t in range(0, len(self.train_times['slices'])):
                test_times_['slices_'] += [
                    _sliding_window(epochs.times, test_times_)]
            test_times_['slices'] = test_times_['slices_']
            del test_times_['slices_']

        # Testing times in milliseconds (only keep last time if multiple time
        # slices)
        test_times_['times_'] = [[epochs.times[t_test[-1]]
                                 for t_test in t_train]
                                 for t_train in test_times_['slices']]
        # Store all testing times parameters
        self.test_times = test_times_

        # Prepare parallel predictions
        parallel, p_time_gen, _ = parallel_func(_predict_time_loop, n_jobs)

        # Loop across estimators (i.e. training times)
        self.y_pred_ = parallel(p_time_gen(X, self.estimators_[t_train], cv,
                                slices, self.predict_mode)
                                for t_train, slices in
                                enumerate(test_times_['slices']))
        return self.y_pred_

    def score(self, epochs=None, y=None):
        """Score Epochs

        Estimate scores across trials by comparing the prediction estimated for
        each trial to its true value.

        Calls ``predict()`` if it has not been already.

        Note. The function updates the ``scorer`` and ``scores_`` attributes.

        Parameters
        ----------
        epochs : instance of Epochs | None, optional
            The epochs. Can be similar to fitted epochs or not.
            If None, it needs to rely on the predictions ``y_pred_``
            generated with ``predict()``.
        y : list | np.ndarray, shape (n_epochs,) | None, optional
            True values to be compared with the predictions ``y_pred_``
            generated with ``predict()`` via ``scorer_``.
            If None and ``predict_mode``=='cross-validation' y = ``y_train_``.
            Defaults to None.

        Returns
        -------
        scores : list of lists of float
            The scores estimated by ``scorer_`` at each training time and each
            testing time (e.g. mean accuracy of ``predict(X)``). Note that the
            number of testing times per training time need not be regular;
            else, np.shape(scores) = [n_train_time, n_test_time].
        """

        from sklearn.metrics import accuracy_score

        # Run predictions if not already done
        if epochs is not None:
            self.predict(epochs)
        else:
            if not hasattr(self, 'y_pred_'):
                raise RuntimeError('Please predict() epochs first or pass '
                                   'epochs to score()')

        # Check scorer
        if self.scorer is None:
            # XXX Need API to identify proper scorer from the clf
            self.scorer = accuracy_score

        # If no regressor is passed, use default epochs events
        if y is None:
            if self.predict_mode == 'cross-validation':
                y = self.y_train_
            else:
                if epochs is not None:
                    y = epochs.events[:, 2]
                else:
                    raise RuntimeError('y is undefined because'
                                       'predict_mode="mean-prediction" and '
                                       'epochs are missing. You need to '
                                       'explicitly specify y.')
            if not np.all(np.unique(y) == np.unique(self.y_train_)):
                raise ValueError('Classes (y) passed differ from classes used '
                                 'for training. Please explicitly pass your y '
                                 'for scoring.')
        elif isinstance(y, list):
            y = np.array(y)
        self.y_true_ = y  # true regressor to be compared with y_pred

        # Preprocessing for parallelization:
        n_jobs = min(len(self.y_pred_[0][0]), check_n_jobs(self.n_jobs))
        parallel, p_time_gen, n_jobs = parallel_func(_score_loop, n_jobs)

        # Score each training and testing time point
        scores = parallel(p_time_gen(self.y_true_, self.y_pred_[t_train],
                                     slices, self.scorer)
                          for t_train, slices
                          in enumerate(self.test_times['slices']))

        self.scores_ = scores
        return scores

    def plot(self, title=None, vmin=None, vmax=None, tlim=None, ax=None,
             cmap='RdBu_r', show=True, colorbar=True,
             xlabel=True, ylabel=True):
        """Plotting function of GeneralizationAcrossTime object

        Plot the score of each classifier at each tested time window.

        Parameters
        ----------
        title : str | None
            Figure title. Defaults to None.
        vmin : float | None
            Min color value for scores. If None, sets to min(gat.scores_).
            Defaults to None.
        vmax : float | None
            Max color value for scores. If None, sets to max(gat.scores_).
            Defaults to None.
        tlim : np.ndarray, (train_min, test_max) | None
            The temporal boundaries. Defaults to None.
        ax : object | None
            Plot pointer. If None, generate new figure. Defaults to None.
        cmap : str | cmap object
            The color map to be used. Defaults to 'RdBu_r'.
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
            Figure title. Defaults to None.
        xmin : float | None, optional
            Min time value. Defaults to None.
        xmax : float | None, optional
            Max time value. Defaults to None.
        ymin : float | None, optional
            Min score value. If None, sets to min(scores). Defaults to None.
        ymax : float | None, optional
            Max score value. If None, sets to max(scores). Defaults to None.
        ax : object | None
            Instance of mataplotlib.axes.Axis. If None, generate new figure.
            Defaults to None.
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
            Figure title. Defaults to None.
        xmin : float | None, optional
            Min time value. Defaults to None.
        xmax : float | None, optional
            Max time value. Defaults to None.
        ymin : float | None, optional
            Min score value. If None, sets to min(scores). Defaults to None.
        ymax : float | None, optional
            Max score value. If None, sets to max(scores). Defaults to None.
        ax : object | None
            Instance of mataplotlib.axes.Axis. If None, generate new figure.
            Defaults to None.
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
            of scorer. Defaults to None.
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


def _predict_time_loop(X, estimators, cv, slices, predict_mode):
    """Aux function of GeneralizationAcrossTime

    Run classifiers predictions loop across time samples.

    Parameters
    ----------
    X : np.ndarray, shape (n_epochs, n_features, n_times)
        To-be-fitted data.
    estimators : arraylike, shape (n_times, n_folds)
        Array of Sklearn classifiers fitted in cross-validation.
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
            if (len(estimators) != cv.n_folds) or (cv.n != Xtrain.shape[0]):
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


def _score_loop(y_true, y_pred, slices, scorer):
    n_time = len(slices)
    # Loop across testing times
    scores = [0] * n_time
    for t, indices in enumerate(slices):
        # Scores across trials
        scores[t] = scorer(y_true, y_pred[t])
    return scores


def _check_epochs_input(epochs, y, picks=None):
    """Aux function of GeneralizationAcrossTime

    Format MNE data into scikit-learn X and y

    Parameters
    ----------
    epochs : instance of Epochs
            The epochs.
    y : np.ndarray shape (n_epochs) | list shape (n_epochs) | None
        To-be-fitted model. If y is None, y == epochs.events.
        Defaults to None.
    picks : array-like of int | None
        Channels to be included. If None only good data channels are used.
        Defaults to None.

    Returns
    -------
    X : np.ndarray, shape (n_epochs, n_selected_chans, n_times)
        To-be-fitted data.
    y : np.ndarray, shape (n_epochs,)
        To-be-fitted model.
    picks : np.ndarray, shape (n_selected_chans,)
        The channels to be used.
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
    x_chunk : np.ndarray, shape (n_epochs, n_features, n_times)
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
    times : np.ndarray, shape (n_times,)
        Array of times from MNE epochs.
    window_params : dict keys: ('start', 'stop', 'step', 'length')
        Either train or test times. See GAT documentation.

    Returns
    -------
    time_pick : list
        List of training slices, indicating for each classifier the time
        sample (in indices of times) to be fitted on.
    """

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

    return time_pick


def _predict(X, estimators):
    """Aux function of GeneralizationAcrossTime

    Predict each classifier. If multiple classifiers are passed, average
    prediction across all classifiers to result in a single prediction per
    classifier.

    Parameters
    ----------
    estimators : np.ndarray, shape (n_folds,) | shape (1,)
        Array of scikit-learn classifiers to predict data.
    X : np.ndarray, shape (n_epochs, n_features, n_times)
        To-be-predicted data
    Returns
    -------
    y_pred : np.ndarray, shape (n_epochs, m_prediction_dimensions)
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
        # initialize predict_results array
        if fold == 0:
            predict_size = _y_pred.shape[1] if _y_pred.ndim > 1 else 1
            y_pred = np.ones((n_epochs, predict_size, n_clf))
        if predict_size == 1:
            y_pred[:, 0, fold] = _y_pred
        else:
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


def _time_gen_one_fold(clf, X, y, train, test, scoring):
    """Aux function of time_generalization"""
    from sklearn.metrics import SCORERS
    n_times = X.shape[2]
    scores = np.zeros((n_times, n_times))
    scorer = SCORERS[scoring]

    for t_train in range(n_times):
        X_train = X[train, :, t_train]
        clf.fit(X_train, y[train])
        for t_test in range(n_times):
            X_test = X[test, :, t_test]
            scores[t_test, t_train] += scorer(clf, X_test, y[test])

    return scores


@deprecated("'time_generalization' will be removed"
            " in MNE v0.10. Use 'GeneralizationAcrossTime' instead.")
@verbose
def time_generalization(epochs_list, clf=None, cv=5, scoring="roc_auc",
                        shuffle=True, random_state=None, n_jobs=1,
                        verbose=None):
    """Fit decoder at each time instant and test at all others

    The function returns the cross-validation scores when the train set
    is from one time instant and the test from all others.

    The decoding will be done using all available data channels, but
    will only work if 1 type of channel is availalble. For example
    epochs should contain only gradiometers.

    Parameters
    ----------
    epochs_list : list of Epochs
        The epochs in all the conditions.
    clf : object | None
        A object following scikit-learn estimator API (fit & predict).
        If None the classifier will be a linear SVM (C=1.) after
        feature standardization.
    cv : integer or cross-validation generator
        If an integer is passed, it is the number of folds (default 5).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects.
    scoring : {string, callable, None}
        A string (see model evaluation documentation in scikit-learn) or
        a scorer callable object / function with signature
        ``scorer(y_true, y_pred)``. Defaults to "roc_auc".
    shuffle : bool
        If True, shuffle the epochs before splitting them in folds.
        Defaults to True.
    random_state : None | int
        The random state used to shuffle the epochs. Ignored if
        shuffle is False. Defaults to None.
    n_jobs : int
        Number of jobs to run in parallel. Each fold is fit
        in parallel. Defaults to 1.

    Returns
    -------
    scores : np.ndarray, shape (n_times, n_times)
        The scores averaged across folds. scores[i, j] contains
        the generalization score when learning at time j and testing
        at time i. The diagonal is the cross-validation score
        at each time-independent instant.

    Notes
    -----
    The function implements the method used in:

    Jean-Remi King, Alexandre Gramfort, Aaron Schurger, Lionel Naccache
    and Stanislas Dehaene, "Two distinct dynamic modes subtend the detection of
    unexpected sounds", PLOS ONE, 2013
    """
    from ..io.pick import channel_type, pick_types
    from sklearn.base import clone
    from sklearn.utils import check_random_state
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_validation import check_cv

    if clf is None:
        scaler = StandardScaler()
        svc = SVC(C=1, kernel='linear')
        clf = Pipeline([('scaler', scaler), ('svc', svc)])

    info = epochs_list[0].info
    data_picks = pick_types(info, meg=True, eeg=True, exclude='bads')

    # Make arrays X and y such that :
    # X is 3d with X.shape[0] is the total number of epochs to classify
    # y is filled with integers coding for the class to predict
    # We must have X.shape[0] equal to y.shape[0]
    X = [e.get_data()[:, data_picks, :] for e in epochs_list]
    y = [k * np.ones(len(this_X)) for k, this_X in enumerate(X)]
    X = np.concatenate(X)
    y = np.concatenate(y)

    cv = check_cv(cv, X, y, classifier=True)

    ch_types = set(channel_type(info, idx) for idx in data_picks)
    logger.info('Running time generalization on %s epochs using %s.'
                % (len(X), ' and '.join(ch_types)))

    if shuffle:
        rng = check_random_state(random_state)
        order = np.argsort(rng.randn(len(X)))
        X = X[order]
        y = y[order]

    parallel, p_time_gen, _ = parallel_func(_time_gen_one_fold, n_jobs)
    scores = parallel(p_time_gen(clone(clf), X, y, train, test, scoring)
                      for train, test in cv)
    scores = np.mean(scores, axis=0)
    return scores
