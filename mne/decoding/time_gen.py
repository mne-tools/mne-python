# Authors: Jean-Remi King <jeanremi.king@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import multiprocessing

import numpy as np
from scipy.stats import mode, rankdata
import matplotlib.pyplot as plt

from mne.parallel import parallel_func
from ..utils import logger, verbose, deprecated
from ..io.pick import channel_type, pick_types


class GeneralizationAcrossTime(object):

    """Create object used to 1) fit a series of classifiers on
    multidimensional time-resolved data, and 2) test the ability of each
    classifier to generalize across other time samples.


    Parameters
    ----------
    clf : object | None
        A object scikit-learn estimator API (fit & predict).
        If None the classifier will be a standard pipeline:
        (scaler, linear SVM (C=1.)).
    cv : int | object, optional, default: 5
        If an integer is passed, it is the number of fold (default 5).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects.
    train_times : dict, optional, default: None
        'slices' : array, shape(n_clfs)
            Array of time slices (in indices) used for each classifier.
        'start' : float
            Time at which to start decoding (in seconds). By default,
            min(epochs.times).
        'stop' : float
            Maximal time at which to stop decoding (in seconds). By default,
            max(times).
        'step' : float
            Duration separating the start of to subsequent classifiers (in
            seconds). By default, equals one time sample.
        'length' : float
            Duration of each classifier (in seconds). By default, equals one
            time sample.
    predict_type : {'predict', 'proba', 'distance'}
        Indicates the type of prediction.
    n_jobs : int
        Number of jobs to run in parallel. Each fold is fit
        in parallel.

    Attributes
    ----------
    y_train_ : np.ndarray, shape(n_samples)
        The categories used for training.
    estimators_ : list of list of sklearn.base.BaseEstimator subclasses.
        The estimators for each time point and each fold.
    y_pred_ : array, shape(n_train_times, n_test_times, n_epochs,
                           n_prediction_dims)
        Class labels for samples in X.
    scores_ : list of lists of float
        The scores (mean accuracy of self.predict(X) wrt. y.).
        It's not an array as the testing times per training time
        need not be regular.
    train_times_ : dict
        The same structure as ``test_times``.

    Notes
    -----
    The function implements the method used in:

    Jean-Remi King, Alexandre Gramfort, Aaron Schurger, Lionel Naccache
    and Stanislas Dehaene, "Two distinct dynamic modes subtend the detection of
    unexpected sounds", PLOS ONE, 2013
    """
    def __init__(self, cv=5, clf=None, train_times=None,
                 predict_type='predict', n_jobs=1):

        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline

        # Store parameters in object
        self.cv = cv
        # Define training sliding window
        self.train_times = {} if train_times is None else train_times

        # Default classification pipeline
        if clf is None:
            scaler = StandardScaler()
            svc = SVC(C=1, kernel='linear')
            clf = Pipeline([('scaler', scaler), ('svc', svc)])
        self.clf = clf
        self.predict_type = predict_type
        self.n_jobs = n_jobs

    def fit(self, epochs, y=None, picks=None):
        """ Train a classifier on each specified time slice.

        Parameters
        ----------
        epochs : instance of Epochs
            The epochs.
        y : array | None, optional, default: None
            To-be-fitted model values. If None, y = [epochs.events[:,2]]
        """
        from sklearn.base import clone
        from sklearn.cross_validation import _check_cv
        n_jobs = self.n_jobs
        # Default channel selection
        if picks is None:
            picks = pick_types(epochs.info, meg=True, eeg=True,
                               exclude='bads')
        # Extract data from MNE structure
        X, y = _check_epochs_input(epochs, y, picks)
        cv = _check_cv(self.cv, X, y, classifier=True)

        self.y_train_ = y

        # Cross validation scheme
        # XXX Cross validation should be transformed into a make_cv, and
        # defined in __init__

        if not 'slices' in self.train_times:
            self.train_times['slices'] = _sliding_window(
                epochs.times, self.train_times)

        # Keep last training times in milliseconds
        t_inds_ = [t[-1] for t in self.train_times['slices']]
        self.train_times['s'] = epochs.times[t_inds_]

        # Chunk X for parallelization
        if n_jobs > 0:
            n_chunk = n_jobs
        else:
            n_chunk = multiprocessing.cpu_count()

        # Parallel across training time
        parallel, p_time_gen, _ = parallel_func(_fit_slices, n_jobs)
        splits = np.array_split(self.train_times['slices'], n_chunk)
        f = lambda x: np.unique(np.concatenate(x))

        out = parallel(p_time_gen(clone(self.clf),
                                  X[..., f(train_slices_chunk)],
                                  y, train_slices_chunk, cv)
                       for train_slices_chunk in splits)
        # Unpack estimators into time slices X folds list of lists.
        self.estimators_ = sum(out, [])
        return self

    def predict(self, epochs, independent=False, test_times=None, picks=None):
        """ Test each classifier on each specified testing time slice.

        Note. This function sets and updates the ``y_pred_`` and the
        ``test_times`` attribute.

        Parameters
        ----------
        epochs : instance of Epochs
            The epochs. Can be similar to fitted epochs or not. See independent
            parameter.
        independent : bool
            Indicates whether data X is independent from the data used to fit
            the  classifier. If independent == True, the predictions from each
            cv fold classifier are averaged. Else, only the prediction from the
            corresponding fold is used.
        test_times : str | dict | None, optional, default: None
            if test_times = 'diagonal', test_times = train_times: decode at
            each time point but does not generalize.
            'slices' : array, shape(n_clfs)
                Array of time slices (in indices) used for each classifier.
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

        Returns
        -------
        y_pred_ : array, shape(n_train_time, n_test_time, n_trials,
                               n_prediction_dim)
            Class labels for samples in X.
        """
        from sklearn.cross_validation import _check_cv
        if picks is None:
            picks = pick_types(epochs.info, meg=True, eeg=True,
                               exclude='bads')
        n_jobs = self.n_jobs
        X, y = _check_epochs_input(epochs, None, picks)
        cv = _check_cv(self.cv, X, y, classifier=True)

        # Check that at least one classifier has been trained
        if not hasattr(self, 'estimators_'):
            raise RuntimeError('Please fit models before trying to predicit')

        # Cross validation scheme: if same data set use CV for prediction, else
        # predict each trial with all folds' classifiers
        self.independent_ = independent  # XXX Good name?

        # Store type of prediction (continuous, categorical etc)

        # Define testing sliding window
        if test_times == 'diagonal':
            test_times_ = {}
            test_times_['slices'] = [[s] for s in self.train_times['slices']]
        elif test_times is None:
            test_times_ = {}
        elif isinstance(test_times, dict):
            test_times_ = test_times
        else:
            raise ValueError('`test_times` must be a dict, "diagonal" or None')
        if not 'slices' in test_times_:
            # Initialize array
            test_times_['slices_'] = []
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
        test_times_['s'] = [[epochs.times[t_test[-1]] for t_test in t_train]
                            for t_train in test_times_['slices']]
        # Store all testing times parameters
        self.test_times_ = test_times_

        # Prepare parallel predictions
        parallel, p_time_gen, _ = parallel_func(_predict_time_loop, n_jobs)

        # Loop across estimators (i.e. training times)
        packed = parallel(p_time_gen(X, self.estimators_[t_train], cv,
                          slices, self.independent_, self.predict_type)
                          for t_train, slices in
                          enumerate(test_times_['slices']))

        self.y_pred_ = np.transpose(zip(*packed), (1, 0, 2, 3))
        return self.y_pred_

    def score(self, epochs, y=None, scorer=None, independent=False,
              test_times=None):
        """ Aux function of GeneralizationAcrossTime
        Estimate score across trials by comparing the prediction estimated for
        each trial to its true value.

        Note. The function updates the ``scores_`` attribute.

        Parameters
        ----------
        epochs : instance of Epochs
            The epochs. Can be similar to fitted epochs or not. See independent
            parameter.
        y : list | array, shape (n_trials) | None, optional, default: None
            To-be-fitted model, If None, y = [epochs.events[:,2]]
        scorer : object
            Sklearn scoring object
        independent : bool
            Indicates whether data X is independent from the data used to fit
            the  classifier. If independent == True, the predictions from each
            cv fold classifier are averaged. Else, only the prediction from the
            corresponding fold is used.
        test_times : str | dict | None, optional, default: None
            if test_times = 'diagonal', test_times = train_times: decode at
            each time point but does not generalize.
            'slices' : array, shape(n_clfs)
                Array of time slices (in indices) used for each classifier.
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

        Returns
        -------
        scores : list of lists of float
            The scores (mean accuracy of self.predict(X) wrt. y.).
            It's not an array as the testing times per training time
            need not be regular.
        """

        from sklearn.metrics import roc_auc_score, accuracy_score

        # Run predictions
        self.predict(epochs, independent=independent, test_times=test_times)

        # If no regressor is passed, use default epochs events
        if y is None:
            if not independent:
                y = self.y_train_  # XXX good name?
            else:
                y = epochs.events[:, 2]
            # make sure it's int
            y = (rankdata(y, 'dense') - 1).astype(np.int)

        self.y_true_ = y  # true regressor to be compared with y_pred

        # Setup scorer
        if scorer is None:
            if self.predict_type == 'predict':
                scorer = accuracy_score
            else:
                scorer = roc_auc_score
        self.scorer_ = scorer

        # Initialize values: Note that this is not an array as the testing
        # times per training time need not be regular
        scores = [[] for i in range(len(self.test_times_['slices']))]

        # Loop across training/testing times
        for t_train, slices in enumerate(self.test_times_['slices']):
            n_time = len(slices)
            # Loop across testing times
            scores[t_train] = [0] * n_time
            for t, indices in enumerate(slices):
                # Scores across trials
                scores[t_train][t] = _score(self.y_true_,
                                            self.y_pred_[t_train][t],
                                            scorer)
        self.scores_ = scores
        return scores

    def plot(self, title=None, vmin=0., vmax=1., tlim=None, ax=None,
             cmap='RdBu_r', show=True):
        """Plotting function of GeneralizationAcrossTime object

        Predict each classifier. If multiple classifiers are passed, average
        prediction across all classifier to result in a single prediction per
        classifier.

        Parameters
        ----------
        title : str | None, optional, default : None
            Figure title.
        vmin : float, optional, default:0.
            Min color value for score.
        vmax : float, optional, default:1.
            Max color value for score.
        tlim : array, (train_min_max, test_min_max) | None, optional,
            default: None
        ax : object | None, optional, default: None
            Plot pointer. If None, generate new figure.
        cmap : str | cmap object
            The color map to be used. Defaults to 'RdBu_r'.
        show : bool, optional, default: True
            plt.show()

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The figure.
        """
        # XXX actually the test seemed wrong and obsolete (D.E.)
        # Check that same amount of testing time per training time
        # assert len(np.unique([len(t) for t in self.test_times_])) == 1
        # Setup plot
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        # Define time limits
        if tlim is None:
            tlim = [self.test_times_['s'][0][0], self.test_times_['s'][-1][-1],
                    self.train_times['s'][0], self.train_times['s'][-1]]
        # Plot scores
        im = ax.imshow(self.scores_, interpolation='nearest', origin='lower',
                       extent=tlim, vmin=vmin, vmax=vmax,
                       cmap=cmap)
        ax.set_xlabel('Testing Time (s)')
        ax.set_ylabel('Training Time (s)')
        if not title is None:
            ax.set_title(title)
        ax.axvline(0, color='k')
        ax.axhline(0, color='k')
        plt.colorbar(im, ax=ax)
        if show:
            plt.show()
        return fig if ax is None else ax.get_figure()

    def plot_diagonal(self, title=None, ymin=0., ymax=1., ax=None, show=True,
                      color='b'):
        """Plotting function of GeneralizationAcrossTime object

        Predict each classifier. If multiple classifiers are passed, average
        prediction across all classifier to result in a single prediction per
        classifier.

        Parameters
        ----------
        title : str | None, optional, default : None
            Figure title.
        ymin : float, optional, default:0.
            Min score value.
        ymax : float, optional, default:1.
            Max score value.
        tlim : array, (train_min_max, test_min_max) | None, optional,
            default: None
        ax : object | None, optional, default: None
            Plot pointer. If None, generate new figure.
        show : bool, optional, default: True
            plt.show()
        color : str, optional, default: 'b'
            Score line color.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The figure.
        """

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        # detect whether gat is a full matrix or just its diagonal
        if np.all(np.unique([len(t) for t in self.test_times_['s']]) == 1):
            scores = self.scores_
        else:
            scores = np.diag(self.scores_)
        ax.plot(self.train_times['s'], scores, color=color,
                label="Classif. score")
        ax.axhline(0.5, color='k', linestyle='--', label="Chance level")
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(self.scorer_.func_name)
        ax.legend(loc='best')
        if show:
            plt.show()
        return fig if ax is None else ax.get_figure()


def _predict_time_loop(X, estimators, cv, slices, independent,
                       predict_type):
    """Aux function of GeneralizationAcrossTime

    Run classifiers predictions loop across time samples.

    Parameters
    ----------
    X : array, shape (n_trials, n_features, n_times)
        To-be-fitted data
    estimators : arraylike, shape(n_times, n_folds)
        Array of Sklearn classifiers fitted in cross-validation.
    slices : list, shape(n_slices)
        List of slices selecting data from X from which is prediction is
        generated.
    independent : bool
        Indicates whether data X is independent from the data used to fit the
        classifier. If independent == True, the predictions from each cv fold
        classifier are averaged. Else, only the prediction from the
        corresponding fold is used.
    predict_type : {'predict', 'proba', 'distance'}
        Indicates the type of prediction .
    """
    n_trial = len(X)
    # Loop across testing slices
    y_pred = [[] for i in range(len(slices))]
    for t, indices in enumerate(slices):
        # Flatten features in case of multiple time samples
        Xtrain = X[:, :, indices].reshape(
            n_trial, np.prod(X[:, :, indices].shape[1:]))

        # Single trial predictions
        if not independent:
            # If predict within cross validation, only predict with
            # corresponding classifier, else predict with each fold's
            # classifier and average prediction.
            for k, (train, test) in enumerate(cv):
                # XXX I didn't manage to initalize correctly this array, as
                # its size depends on the the type of predicter and the
                # number of class.
                if k == 0:
                    y_pred_ = _predict(Xtrain[test, :],
                                       estimators[k:k + 1], predict_type)
                    y_pred[t] = np.empty((n_trial, y_pred_.shape[1]))
                    y_pred[t][test, :] = y_pred_
                y_pred[t][test, :] = _predict(Xtrain[test, :],
                                              estimators[k:k + 1],
                                              predict_type)
        else:
            y_pred[t] = _predict(Xtrain, estimators, predict_type)
    return y_pred


def _score(y, y_pred, scorer):
    """Aux function of GeneralizationAcrossTime

    Estimate classifiaction score.

    Parameters
    ----------
    y : list | array, shape (n_trials)
        True model value
    y_pred : list | array, shape (n_trials)
        Classifier prediction of model value
    scorer : object
        scikit-learn scoring object

    Returns
    -------
    score : float
        Score estimated across all trials for each train/tested time sample.
    y_pred : array, shape(n_slices, n_trials)
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
        score = 0
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
    y : array shape(n_trials) | list shape(n_trials) | None
        To-be-fitted model. If y is None, y == epochs.events
    picks : array (n_selected_chans) | None
        Channels to be included in scikit-learn model fitting.

    Returns
    -------
    X : array, shape(n_trials, n_selected_chans, n_times)
        To-be-fitted data
    y : array, shape(n_trials)
        To-be-fitted model
    picks : array, shape()
    """
    if y is None:
        y = epochs.events[:, 2]
        y = (rankdata(y, 'dense') - 1).astype(np.int)

    # Convert MNE data into trials x features x time matrix
    X = epochs.get_data()[:, picks, :]
    # Check data sets
    assert X.shape[0] == y.shape[0]
    return X, y


def _fit_slices(clf, Xchunk, y, slices, cv):
    """Aux function of GeneralizationAcrossTime

    Fit each classifier.

    Parameters
    ----------
    clf : scikit-learn classifier
    Xchunk : array, shape (n_trials, n_features, n_times)
        To-be-fitted data
    y : list | array, shape (n_trials)
        To-be-fitted model
    slices : list | array, shape(n_training_slice)
        List of training slices, indicating time sample relative to X
    cv : scikit-learn cross-validater

    Returns
    -------
    estimators : list
        List of fitted Sklearn classifiers corresponding to each training
        slice.
    """
    from sklearn.base import clone
    # Initialize
    n_trials = len(Xchunk)
    estimators = []
    # Identify the time samples of X_chunck corresponding to X
    values = np.unique(np.concatenate(slices))
    indices = range(len(values))
    # Loop across time slices
    for t_slice in slices:
        # Translate absolute time samples into time sample relative to Xchunk
        for ii in indices:
            t_slice[t_slice == values[ii]] = indices[ii]
        # Select slice
        X = Xchunk[..., t_slice]
        # Reshape data matrix to flatten features in case of multiple time
        # samples.
        X = X.reshape(n_trials, np.prod(X.shape[1:]))
        # Loop across folds
        estimators_ = []
        for fold, (train, test) in enumerate(cv):
            # Fit classifier
            clf_ = clone(clf)
            clf_.fit(X[train, :], y[train])
            estimators_.append(clf_)
        # Store classifier
        estimators.append(estimators_)
    return estimators


def _sliding_window(times, tt_times):
    """Aux function of GeneralizationAcrossTime

    Define the slices on which to train each classifier.

    Parameters
    ----------
    times : array, shape (n_times)
        Array of times from MNE epochs
    tt_times : dict, optional keys: ('start', 'stop', 'step', 'length' )
        Either train or test times. See GAT documentation.

    Returns
    -------
    time_pick : list, shape(n_classifiers)
        List of training slices, indicating for each classifier the time
        sample (in indices of times) to be fitted on.
    """

    # Sampling frequency as int
    freq = (times[-1] - times[0]) / len(times)

    # Default values
    if ('slices' in tt_times and
            all(k in tt_times for k in ('start', 'stop', 'step', 'length'))):
        time_pick = tt_times['slices']
    else:
        if not 'start' in tt_times:
            tt_times['start'] = times[0]
        if not 'stop' in tt_times:
            tt_times['stop'] = times[-1]
        if not 'step' in tt_times:
            tt_times['step'] = freq
        if not 'length' in tt_times:
            tt_times['length'] = freq

        # Convert seconds to index

        def find_time_idx(t):  # find closest time point
            return np.argmin(np.abs(np.asarray(times) - t))

        start = find_time_idx(tt_times['start'])
        stop = find_time_idx(tt_times['stop'])
        step = int(round(tt_times['step'] / freq))
        length = int(round(tt_times['length'] / freq))

        # For each training slice, give time samples to be included
        time_pick = [range(start, start + length)]
        while (time_pick[-1][0] + step) <= (stop - length + 1):
            start = time_pick[-1][0] + step
            time_pick.append(range(start, start + length))

    return time_pick


def _predict(X, estimators, predict_type):
    """Aux function of GeneralizationAcrossTime

    Predict each classifier. If multiple classifiers are passed, average
    prediction across all classifier to result in a single prediction per
    classifier.

    Parameters
    ----------
    estimators : array, shape(n_folds) or shape(1)
        Array of scikit-learn classifiers to predict data
    X : array, shape (n_trials, n_features, n_times)
        To-be-predicted data
    predict_type : {'predict', 'distance', 'proba'}
        'predict' => simple prediction of y (e.g. SVC, SVR)
        'distance' => continuous prediction (e.g. decision_function)
        'proba' => probabilistic prediction (e.g. SVC(probability=True))

    Returns
    -------
    y_pred : array, shape(n_trials, m_prediction_dimensions)
        Classifier's prediction for each trial.
    """
    # Initialize results:
    # XXX Here I did not manage to find an efficient and generic way to guess
    # the number of output provided by predict, and could thus not initalize
    # the y_pred values.
    n_trial = X.shape[0]
    n_clf = len(estimators)
    if predict_type == 'predict':
        n_class = 1
    elif predict_type == 'distance':
        dec_func = estimators[0].decision_function(X[0, :])
        if len(dec_func.shape) > 1:
            n_class = dec_func.shape[1]
        else:  # certain binary cases for which output is raveled.
            n_class = 2

    elif predict_type == 'proba':
        n_class = estimators[0].predict_proba(X[0, :]).shape[1]
    y_pred = np.ones((n_trial, n_class, n_clf))

    # Compute prediction for each sub-estimator (i.e. per fold)
    # if independent, estimators = all folds
    for fold, clf in enumerate(estimators):
        if predict_type == 'predict':
            # Discrete categorical prediction
            y_pred[:, 0, fold] = clf.predict(X)
        elif predict_type == 'proba':
            # Probabilistic prediction
            y_pred[:, :, fold] = clf.predict_proba(X)
        elif predict_type == 'distance':
            # Continuous non-probabilistic predict
            # XXX branching fixes test for binary cases.
            dec_func = clf.decision_function(X)
            if len(dec_func.shape) > 1:
                y_pred[:, :, fold] = dec_func
            else:
                y_pred[:, 0, fold] = dec_func

    # Collapse y_pred across folds if necessary (i.e. if independent)
    if fold > 0:
        if predict_type == 'predict':
            y_pred, _ = mode(y_pred, axis=2)
        else:
            y_pred = np.mean(y_pred, axis=2)

    # Remove unnecessary symetrical prediction (i.e. for probas & distances)
    if predict_type != 'predict' and y_pred.shape[1] == 2:
        y_pred = y_pred[:, 1, :]
        n_class = 1

    # Format shape
    y_pred = y_pred.reshape((n_trial, n_class))
    return y_pred


def _time_loop(clf, scorer, X, y, train, test, X_gen, y_gen, train_slice,
               test_slices):
    # Initialize results
    scores = []  # scores
    tested = []  # tested time points
    scores_gen = []  # generalization score
    # Flatten features
    my_reshape = lambda X: X.reshape(len(X), np.prod(X.shape[1:]))
    # Select training set
    X_train = my_reshape(X[train, :, train_slice])
    # Fit classifier
    clf.fit(X_train, y[train])
    # Test classification performance across testing time
    for test_slice in test_slices:
        # Select testing time slice
        X_test = my_reshape(X[test, :, test_slice])
        # Evaluate classifer on cross-validation set
        # and store result in relative sampling-time
        scores.append(scorer(clf, X_test, y[test]))
        tested.append(True)
        # Evaluate classifier on cross-condition generalization set
        if X_gen is not None and y_gen is not None:
            x_gen = my_reshape(X_gen[..., test_slice])
            scores_gen.append(scorer(clf, x_gen, y_gen))

    return scores, scores_gen, tested


def _compress_results(scores, tested):
    """"
    Avoids returning partially empty results by removing empty lines and
    columns (generally due to slice length > 1).
    """
    scores = scores[:, np.any(tested, axis=0)]
    scores = scores[np.any(tested, axis=1), :]
    return scores


def _gen_type(n_samples, relative_test_slice=False, train_slices=None,
              test_slices=None):
    """ Creates typical temporal generalization scenarios

    The function return train_slices, test_slices that indicate the time
    samples to be used for training and testing each classifier. These
    lists can be directly used by time_generalization_Xy()

    Parameters
    ----------
    n_samples : int
        Number of time samples in each trial | Last sample to on which the
        classifier can be trained
    relative_test_slice : bool
        True implies that the samples indicated in test_slices are relative to
        the samples in train_slices. False implies that the samples in
        test_slices corresponds to the actual data samples.
    train_slices : list | callable | None
        List of slices generated with create_slices(). By default the
        classifiers are trained on all time points (i.e.
        create_slices(n_time)).
    test_slices : list |  callable | None
        List of slices generated with create_slices(). By default the
        classifiers are tested on all time points (i.e.
        [create_slices(n_time)] * n_time).
    """
    from ..utils import create_slices  # To be deprecated in v0.10

    # Setup train slices
    if train_slices is None:
        # default: train and test over all time samples
        train_slices = create_slices(0, n_samples)
    elif callable(train_slices):
        # create slices once n_slices is known
        train_slices = train_slices(0, n_samples)

    # Setup test slices
    if not relative_test_slice:
        # Time generalization is from/to particular time samples
        if test_slices is None:
            # Default: testing time is identical to training time
            test_slices = [train_slices] * len(train_slices)
        elif callable(test_slices):
            test_slices = [test_slices(n_samples)] * len(train_slices)

    else:
        # Time generalization is at/around the training time samples
        if test_slices is None:
            # Default: testing times are identical to training slices
            # (classic decoding across time)
            test_slices = [[s] for s in train_slices]
        else:
            # Update slice by combining timing of test and train slices
            up_slice = lambda test, train: slice(test.start + train.start,
                                                 test.stop + train.stop - 1,
                                                 train.step)

            test_slices = np.tile(
                [test_slices], (len(train_slices), 1)).tolist()
            for t_train in range(len(train_slices)):
                for t_test in range(len(test_slices[t_train])):
                    # Add start and stop of training and testing slices
                    # to make testing timing dependent on training timing
                    test_slices[t_train][t_test] = up_slice(
                        test_slices[t_train][t_test],
                        train_slices[t_train])

    # Check that all time samples are in bounds
    if any([(s.start < 0) or (s.stop > n_samples) for s in train_slices]) or \
       any([(s.start < 0) or (s.stop > n_samples) for ss in test_slices
            for s in ss]):
        logger.warn('Slicing: time samples out of bound!')
        # Shortcut to select slices that are in bounds
        sel = lambda slices, bol: [s for s, b in zip(slices, bol) if b]

        # Deal with testing slices first:
        for t_train in range(len(test_slices)):
            # Find testing slices that are in bounds
            inbound = [(s.start >= 0) and (s.stop <= n_samples)
                       for s in test_slices[t_train]]
            test_slices[t_train] = sel(test_slices[t_train], inbound)

        # Deal with training slices then:
        inbound = [(s.start >= 0) and (s.stop <= n_samples)
                   for s in train_slices]
        train_slices = sel(train_slices, inbound)

    return train_slices, test_slices


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
    cv : integer or cross-validation generator, optional
        If an integer is passed, it is the number of fold (default 5).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects.
    scoring : {string, callable, None}, optional, default: "roc_auc"
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
    scores : array, shape (n_times, n_times)
        The scores averaged across folds. scores[i, j] contains
        the generalization score when learning at time j and testing
        at time i. The diagonal is the cross-validation score
        at each time-independant instant.

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
