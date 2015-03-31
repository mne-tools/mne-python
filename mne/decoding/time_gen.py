# Authors: Jean-Remi King <jeanremi.king@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Clement Moutard <clement.moutard@gmail.com>
#
# License: BSD (3-clause)

import multiprocessing

import numpy as np
import warnings
from scipy import stats

from ..viz.decoding import plot_gat_matrix, plot_gat_diagonal
from ..parallel import parallel_func
from ..utils import logger, verbose, deprecated
from ..io.pick import channel_type, pick_types


class _DecodingTime(dict):
    """A dictionary to configure the training times that has the following keys:

    'slices' : np.ndarray, shape (n_clfs,)
        Array of time slices (in indices) used for each classifier.
        If not given, computed from 'start', 'stop', 'length', 'step'.
    'start' : float
        Time at which to start decoding (in seconds). By default,
        min(epochs.times).
    'stop' : float
        Maximal time at which to stop decoding (in seconds). By default,
        max(times).
    'step' : float
        Duration separating the start of subsequent classifiers (in
        seconds). By default, equals one time sample.
    'length' : float
        Duration of each classifier (in seconds). By default, equals one
        time sample.
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
    clf : object | None
        An estimator compliant with the scikit-learn API (fit & predict).
        If None the classifier will be a standard pipeline including
        StandardScaler and a linear SVM with default parameters.
    cv : int | object
        If an integer is passed, it is the number of folds.
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects.
        Defaults to 5.
    train_times : dict | None
        A dictionary to configure the training times.
        'slices' : np.ndarray, shape (n_clfs,)
            Array of time slices (in indices) used for each classifier.
            If not given, computed from 'start', 'stop', 'length', 'step'.
        'start' : float
            Time at which to start decoding (in seconds). By default,
            min(epochs.times).
        'stop' : float
            Maximal time at which to stop decoding (in seconds). By default,
            max(times).
        'step' : float
            Duration separating the start of subsequent classifiers (in
            seconds). By default, equals one time sample.
        'length' : float
            Duration of each classifier (in seconds). By default, equals one
            time sample.
        If None, empty dict. Defaults to None.
    predict_type : {'predict', 'predict_proba', 'decision_function'}
        Indicates the type of prediction:
            'predict' : generates a categorical estimate of each trial.

            'predict_proba' : generates a probabilistic estimate of each trial.

            'decision_function' : generates a continuous non-probabilistic
                estimate of each trial.
        Default: 'predict'
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
    n_jobs : int
        Number of jobs to run in parallel. Defaults to 1.

    Attributes
    ----------
    y_train_ : np.ndarray, shape (n_samples,)
        The categories used for training.
    estimators_ : list of list of sklearn.base.BaseEstimator subclasses.
        The estimators for each time point and each fold.
    y_pred_ : np.ndarray, shape (n_train_times, n_test_times, n_epochs,
                           n_prediction_dims)
        Class labels for samples in X.
    scores_ : list of lists of float
        The scores (mean accuracy of self.predict(X) wrt. y.).
        It's not an array as the testing times per training time
        need not be regular.
    test_times_ : dict
        The same structure as ``train_times``.
    cv_ : CrossValidation object
        The actual CrossValidation input depending on y.

    Notes
    -----
    The function implements the method used in:

    Jean-Remi King, Alexandre Gramfort, Aaron Schurger, Lionel Naccache
    and Stanislas Dehaene, "Two distinct dynamic modes subtend the detection of
    unexpected sounds", PLOS ONE, 2013
    """
    def __init__(self, cv=5, clf=None, train_times=None,
                 predict_type='predict', predict_mode='cross-validation',
                 n_jobs=1):

        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline

        # Store parameters in object
        self.cv = cv
        # Define training sliding window
        self.train_times = (_DecodingTime() if train_times is None
                            else _DecodingTime(train_times))

        # Default classification pipeline
        if clf is None:
            scaler = StandardScaler()
            svc = SVC(C=1, kernel='linear')
            clf = Pipeline([('scaler', scaler), ('svc', svc)])
        self.clf = clf
        self.predict_type = predict_type
        self.predict_mode = predict_mode
        self.n_jobs = n_jobs

    def __repr__(self):
        s = ''
        if hasattr(self, "estimators_"):
            s += "fitted, start : %0.3f (s), stop : %0.3f (s)" % (
                self.train_times['start'], self.train_times['stop'])
        else:
            s += 'no fit'
        if hasattr(self, 'y_pred_'):
            s += (", predict_type : '%s' on %d epochs" % (
                self.predict_type, len(self.y_pred_)))
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

    def fit(self, epochs, y=None, picks=None):
        """ Train a classifier on each specified time slice.

        Parameters
        ----------
        epochs : instance of Epochs
            The epochs.
        y : np.ndarray of int, shape (n_samples,) | None
            To-be-fitted model values. If None, y = epochs.events[:, 2].
            Defaults to None.
        picks : np.ndarray of int, shape (n_channels,) | None
            The channels to be used. If None, defaults to meg and eeg channels.
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
        n_jobs = self.n_jobs
        # Default channel selection
        if picks is None:
            picks = pick_types(epochs.info, meg=True, eeg=True,
                               exclude='bads')
        # Extract data from MNE structure
        X, y = _check_epochs_input(epochs, y, picks)
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

        # Chunk X for parallelization
        if n_jobs > 0:
            n_chunk = n_jobs
        else:
            n_chunk = multiprocessing.cpu_count()

        # Avoid splitting the data in more time chunk than there is time points
        if n_chunk > X.shape[2]:
            n_chunk = X.shape[2]

        # Parallel across training time
        parallel, p_time_gen, n_jobs = parallel_func(_fit_slices, n_jobs)
        splits = np.array_split(self.train_times['slices'], n_chunk)

        def f(x):
            return np.unique(np.concatenate(x))

        out = parallel(p_time_gen(clone(self.clf),
                                  X[..., f(train_slices_chunk)],
                                  y, train_slices_chunk, cv)
                       for train_slices_chunk in splits)
        # Unpack estimators into time slices X folds list of lists.
        self.estimators_ = sum(out, list())
        return self

    def predict(self, epochs, test_times=None, picks=None):
        """ Test each classifier on each specified testing time slice.

        Note. This function sets and updates the ``y_pred_`` and the
        ``test_times`` attribute.

        Parameters
        ----------
        epochs : instance of Epochs
            The epochs. Can be similar to fitted epochs or not. See independent
            parameter.
        test_times : str | dict | None
            A dict to configure the testing times.
            If test_times = 'diagonal', test_times = train_times: decode at
            each time point but does not generalize.

            'slices' : np.ndarray, shape (n_clfs,)
                Array of time slices (in indices) used for each classifier.
                If not given, computed from 'start', 'stop', 'length', 'step'.
            'start' : float
                Time at which to start decoding (in seconds). By default,
                min(epochs.times).
            'stop' : float
                Maximal time at which to stop decoding (in seconds). By
                default, max(times).
            'step' : float
                Duration separating the start of to subsequent classifiers (in
                seconds). By default, equals one time sample.
            'length' : float
                Duration of each classifier (in seconds). By default, equals
                one time sample.

            If None, empty dict. Defaults to None.

        Returns
        -------
        y_pred_ : np.ndarray, shape (n_train_time, n_test_time, n_epochs,
                               n_prediction_dim)
            Class labels for samples in X.
        """
        if picks is None:
            picks = pick_types(epochs.info, meg=True, eeg=True,
                               exclude='bads')
        n_jobs = self.n_jobs
        X, y = _check_epochs_input(epochs, None, picks)

        # Check that at least one classifier has been trained
        if not hasattr(self, 'estimators_'):
            raise RuntimeError('Please fit models before trying to predict')
        cv = self.cv_  # Retrieve CV scheme from fit()

        # Define testing sliding window
        if test_times == 'diagonal':
            test_times_ = _DecodingTime()
            test_times_['slices'] = [[s] for s in self.train_times['slices']]
        elif test_times is None:
            test_times_ = _DecodingTime()
        elif isinstance(test_times, dict):
            test_times_ = test_times
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
        self.test_times_ = test_times_

        # Prepare parallel predictions
        parallel, p_time_gen, _ = parallel_func(_predict_time_loop, n_jobs)

        # Loop across estimators (i.e. training times)
        packed = parallel(p_time_gen(X, self.estimators_[t_train], cv,
                          slices, self.predict_mode, self.predict_type)
                          for t_train, slices in
                          enumerate(test_times_['slices']))

        self.y_pred_ = np.transpose(tuple(zip(*packed)), (1, 0, 2, 3))
        return self.y_pred_

    def score(self, epochs=None, y=None, scorer=None, test_times=None):
        """Score Epochs

        Estimate scores across trials by comparing the prediction estimated for
        each trial to its true value.

        Note. The function updates the ``scores_`` attribute.

        Parameters
        ----------
        epochs : instance of Epochs | None
            The epochs. Can be similar to fitted epochs or not. See independent
            parameter.
            If None, it relies on the y_pred_ generated from predit()
        y : list | np.ndarray, shape (n_epochs,) | None
            To-be-fitted model, If None, y = epochs.events[:,2].
            Defaults to None.
        scorer : object
            scikit-learn Scorer instance.
        test_times : str | dict | None
            if test_times = 'diagonal', test_times = train_times: decode at
            each time point but does not generalize. If dict, the following
            structure is expected.

            'slices' : np.ndarray, shape (n_clfs,)
                Array of time slices (in indices) used for each classifier.
                If not given, computed from 'start', 'stop', 'length', 'step'.
            'start' : float
                Time at which to start decoding (in seconds). By default,
                min(epochs.times).
            'stop' : float
                Maximal time at which to stop decoding (in seconds). By
                default, max(times).
            'step' : float
                Duration separating the start of to subsequent classifiers (in
                seconds). By default, equals one time sample.
            'length' : float
                Duration of each classifier (in seconds). By default, equals
                one time sample.

            If None, empty dict. Defaults to None.

        Returns
        -------
        scores : list of lists of float
            The scores (mean accuracy of self.predict(X) wrt. y.).
            It's not an array as the testing times per training time
            need not be regular.
        """

        from sklearn.metrics import (roc_auc_score, accuracy_score,
                                     mean_squared_error)
        from sklearn.base import is_classifier
        from sklearn.preprocessing import LabelEncoder

        # Run predictions if not already done
        if epochs is not None:
            self.predict(epochs, test_times=test_times)
        else:
            if not hasattr(self, 'y_pred_'):
                raise RuntimeError('Please predit() epochs first or pass '
                                   'epochs to score()')

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
        self.y_true_ = y  # true regressor to be compared with y_pred

        # Setup default scorer
        if scorer is None:
            if is_classifier(self.clf):  # Classification
                if self.predict_type == 'predict':
                    # By default, use accuracy if categorical prediction
                    scorer = accuracy_score
                else:
                    # By default, use AUC for continuous output
                    scorer = roc_auc_score
            else:  # Regression  XXX ideally, would need an is_regresser()
                scorer = mean_squared_error
        self.scorer_ = scorer

        # Identify training classes
        training_classes = np.unique(self.y_train_)
        # Change labels if decision_function with inadequate categorical
        # classes
        if (self.predict_type == 'decision_function' and
            not np.array_equiv(training_classes,
                               np.arange(0, len(training_classes))) and
                is_classifier(self.clf)):
            warnings.warn('Scoring categorical predictions from '
                          '`decision_function` requires specific labeling. '
                          'Prefer using a predefined label scheme with '
                          '`sklearn.preprocessing.LabelEncoder`.')
            # set sklearn Label encoder
            le = LabelEncoder()
            le.fit(training_classes)
            transform_classes = True
        else:
            transform_classes = False

        # Initialize values: Note that this is not an array as the testing
        # times per training time need not be regular
        scores = [list() for _ in range(len(self.test_times_['slices']))]

        # Loop across training/testing times
        for t_train, slices in enumerate(self.test_times_['slices']):
            n_time = len(slices)
            # Loop across testing times
            scores[t_train] = [0] * n_time
            for t, indices in enumerate(slices):
                y_true = self.y_true_
                y_pred = self.y_pred_[t_train][t]
                # Transform labels for Sklearn API compatibility
                if transform_classes:
                    y_true = le.transform(y_true)
                # Scores across trials
                scores[t_train][t] = _score(y_true, y_pred, scorer)
        self.scores_ = scores
        return scores

    def plot(self, title=None, vmin=0., vmax=1., tlim=None, ax=None,
             cmap='RdBu_r', show=True, colorbar=True,
             xlabel=True, ylabel=True):
        """Plotting function of GeneralizationAcrossTime object

        Predict each classifier. If multiple classifiers are passed, average
        prediction across all classifiers to result in a single prediction per
        classifier.

        Parameters
        ----------
        title : str | None
            Figure title. Defaults to None.
        vmin : float
            Min color value for score. Defaults to 0..
        vmax : float
            Max color value for score. Defaults to 1.
        tlim : np.ndarray, (train_min, test_max) | None
            The temporal boundaries. defaults to None.
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

    def plot_diagonal(self, title=None, xmin=None, xmax=None, ymin=0., ymax=1.,
                      ax=None, show=True, color='steelblue', xlabel=True,
                      ylabel=True, legend=True):
        """Plotting function of GeneralizationAcrossTime object

        Predict each classifier. If multiple classifiers are passed, average
        prediction across all classifiers to result in a single prediction per
        classifier.

        Parameters
        ----------
        title : str | None
            Figure title. Defaults to None.
        xmin : float | None, optional, defaults to None.
            Min time value.
        xmax : float | None, optional, defaults to None.
            Max time value.
        ymin : float
            Min score value. Defaults to 0.
        ymax : float
            Max score value. Defaults to 1.
        ax : object | None
            Instance of mataplotlib.axes.Axis. If None, generate new figure.
            Defaults to None.
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

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The figure.
        """
        return plot_gat_diagonal(self, title=title, xmin=xmin, xmax=xmax,
                                 ymin=ymin, ymax=ymax, ax=ax, show=show,
                                 color=color, xlabel=xlabel, ylabel=ylabel,
                                 legend=legend)


def _predict_time_loop(X, estimators, cv, slices, predict_mode, predict_type):
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
    predict_type : {'predict', 'predict_proba', 'decision_function'}
        Indicates the type of prediction:
            'predict' : generates a categorical estimate of each trial.
            'predict_proba' : generates a probabilistic estimate of each trial.
            'decision_function' : generates a continuous non-probabilistic
                estimate of each trial.
        Default: 'predict'
    predict_mode :{'cross-validation', 'mean-prediction'}
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
                    y_pred_ = _predict(Xtrain[test, :],
                                       estimators[k:k + 1], predict_type)
                    y_pred[t] = np.empty((n_epochs, y_pred_.shape[1]))
                    y_pred[t][test, :] = y_pred_
                y_pred[t][test, :] = _predict(Xtrain[test, :],
                                              estimators[k:k + 1],
                                              predict_type)
        elif predict_mode == 'mean-prediction':
            y_pred[t] = _predict(Xtrain, estimators, predict_type)
        else:
            raise ValueError('`predict_mode` must be a str, "mean-prediction"'
                             ' or "cross-validation"')
    return y_pred


def _score(y, y_pred, scorer):
    """Aux function of GeneralizationAcrossTime

    Estimate classification score.

    Parameters
    ----------
    y : list | np.ndarray, shape (n_epochs,)
        True model value
    y_pred : list | np.ndarray, shape (n_epochs,)
        Classifier prediction of model value.
    scorer : scorer object
        scikit-learn scoring object.

    Returns
    -------
    score : float
        Score estimated across all trials for each train/tested time sample.
    y_pred : np.ndarray, shape (n_slices, n_epochs)
        Single trial prediction for each train/tested time sample.
    """
    classes = np.unique(y)
    # if binary prediction or discrete prediction
    if y_pred.shape[1] == 1:
        # XXX Problem here with scorer when proba=True but y !=  (0 | 1).
        # Bug Sklearn?
        score = scorer(y, y_pred)
    else:
        # XXX This part is not sufficiently generic to apply to all
        # classification and regression cases.
        score = 0.
        for ii, c in enumerate(classes):
            score += scorer(y == c, y_pred[:, ii])
        score /= len(classes)
    return score


def _check_epochs_input(epochs, y, picks):
    """Aux function of GeneralizationAcrossTime

    Format MNE data into scikit-learn X and y

    Parameters
    ----------
    epochs : instance of Epochs
            The epochs.
    y : np.ndarray shape (n_epochs) | list shape (n_epochs) | None
        To-be-fitted model. If y is None, y == epochs.events
    picks : np.ndarray (n_selected_chans,) | None
        Channels to be included in scikit-learn model fitting.

    Returns
    -------
    X : np.ndarray, shape (n_epochs, n_selected_chans, n_times)
        To-be-fitted data.
    y : np.ndarray, shape (n_epochs,)
        To-be-fitted model.
    picks : np.ndarray, shape (n_channels,)
        The channels to be used.
    """
    if y is None:
        y = epochs.events[:, 2]

    # Convert MNE data into trials x features x time matrix
    X = epochs.get_data()[:, picks, :]
    # Check data sets
    assert X.shape[0] == y.shape[0]
    return X, y


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


def _predict(X, estimators, predict_type):
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
    predict_type : str, {'predict', 'predict_proba',
                  'decision_function'}
        Indicates the type of prediction:
            'predict' : generates a categorical estimate of each trial.
            'predict_proba' : generates a probabilistic estimate of each trial.
            'decision_function' : generates a continuous non-probabilistic
                estimate of each trial.
        Default: 'predict'
    Returns
    -------
    y_pred : np.ndarray, shape (n_epochs, m_prediction_dimensions)
        Classifier's prediction for each trial.
    """
    # Initialize results:
    # XXX Here I did not manage to find an efficient and generic way to guess
    # the number of output provided by predict, and could thus not initalize
    # the y_pred values.
    n_epochs = X.shape[0]
    n_clf = len(estimators)
    n_class = 1  # initialize
    if predict_type == 'predict':
        n_class = 1
    elif predict_type == 'predict_proba':
        n_class = estimators[0].predict_proba(X[0, :]).shape[-1]
    elif predict_type == 'decision_function':
        shape = estimators[0].decision_function(X[0, :]).shape
        if len(shape) > 1:  # deal with sklearn APIs
            n_class = shape[1]

    else:
        raise ValueError('predict_type must be "predict" or "predict_proba" '
                         'or "decision_function"')
    y_pred = np.ones((n_epochs, n_class, n_clf))

    # Compute prediction for each sub-estimator (i.e. per fold)
    # if independent, estimators = all folds
    for fold, clf in enumerate(estimators):
        if predict_type == 'predict':
            # Discrete categorical prediction
            y_pred[:, 0, fold] = clf.predict(X)
        elif predict_type == 'predict_proba':
            # Probabilistic prediction
            y_pred[:, :, fold] = clf.predict_proba(X)
        elif predict_type == 'decision_function':
            # continuous non-probabilistic prediction
            y_ = clf.decision_function(X)
            if y_.ndim == 1:  # new sklearn versions seem to return 1d arrays
                y_ = y_[:, None]
            y_pred[:, :, fold] = y_

    # Collapse y_pred across folds if necessary (i.e. if independent)
    if fold > 0:
        if predict_type == 'predict':
            y_pred, _ = stats.mode(y_pred, axis=2)
        else:
            y_pred = np.mean(y_pred, axis=2)

    # Format shape
    y_pred = y_pred.reshape((n_epochs, n_class))
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
    scoring : {string, callable, None}, default: "roc_auc"
        A string (see model evaluation documentation in scikit-learn) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    shuffle : bool
        If True, shuffle the epochs before splitting them in folds.
    random_state : None | int
        The random state used to shuffle the epochs. Ignored if
        shuffle is False.
    n_jobs : int
        Number of jobs to run in parallel. Each fold is fit
        in parallel.

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
    logger.info('Running time generalization on %s epochs using %s.' %
                (len(X), ' and '.join(ch_types)))

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
