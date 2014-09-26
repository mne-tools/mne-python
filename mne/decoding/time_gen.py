# Authors: Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt

from mne.parallel import parallel_func
from mne import pick_types

from sklearn.cross_validation import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


class GeneralizationAcrossTime(object):

    """Create object used to 1) fit a series of classifiers on 
    multidimensional time-resolved data, and 2) test the ability of each 
    classifier to generalize across other time samples.
    

    Parameters
    ----------
    clf : object | None
        A object Scikit-Learn estimator API (fit & predict).
        If None the classifier will be a standard pipeline:
        (scaler, linear SVM (C=1.)).
    cv : int | object, optional, default: 5
        If an integer is passed, it is the number of fold (default 5).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects.
    chans_picks : array (n_selected_chans) | None, optional, default: None
        Channels to be included in Sklearn model fitting.
    train_times : dict, optional, default: {} 
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
            Duration of each classifier (in seconds). By default, equals one time sample.
    n_jobs : int
        Number of jobs to run in parallel. Each fold is fit
        in parallel.
    parallel_across : str, 'folds' | 'time_samples'
        Set the parallel (multi-core) computation across folds or across
        time samples.

    Returns
    -------
    gat : object
        gat.fit() is used to train classifiers
        gat.predict() is used to test the classifiers on existing or novel data

    Notes
    -----
    The function implements the method used in:

    Jean-Remi King, Alexandre Gramfort, Aaron Schurger, Lionel Naccache
    and Stanislas Dehaene, "Two distinct dynamic modes subtend the detection of
    unexpected sounds", PLOS ONE, 2013
    """

    def __init__(self, cv=5, clf=None,
                 chans_picks=None,
                 train_times={},
                 parallel_across='folds', n_jobs=1):

        # Store parameters in object
        self.cv = cv
        self.clf = clf
        self.n_jobs = n_jobs
        self.chans_picks = chans_picks
        self.train_times = train_times
        self.parallel_across = parallel_across
        self.chans_picks = chans_picks

        # Default classification pipeline
        if clf is None:
            scaler = StandardScaler()
            svc = SVC(C=1, kernel='linear')
            clf = Pipeline([('scaler', scaler), ('svc', svc)])
        self.clf = clf

    def fit(self, epochs, y=None):
        """ Train a classifier on each specified time slice.
        
        Parameters
        ----------
        epochs : object
            MNE epochs
        y : array | None, optional, default: None
            To-be-fitted model values. If None, y = [epochs.events[:,2]]

        """
        # Default channel selection
        # /!\ Channel selection should be transformed into a make_chans_pick,
        # and defined in __init__
        if self.chans_picks is None:
            info = epochs.info
            self.chans_picks = pick_types(
                info, meg=True, eeg=True, exclude='bads')

        # Extract data from MNE structure
        X, y = _format_data(epochs, y, self.chans_picks)

        # Cross validation scheme
        # /!\ Cross validation should be transformed into a make_cv, and
        # defined in __init__
        from sklearn.cross_validation import check_cv
        if self.cv.__class__ == int:
            self.cv = StratifiedKFold(y, self.cv)
        self.cv = check_cv(self.cv, X, y, classifier=True)

        # Define training sliding window
        self.train_times['slices'] = _sliding_window(
            epochs.times, self.train_times)

        # Keep last training times in milliseconds
        self.train_times['s'] = epochs.times[[t[-1]
            for t in self.train_times['slices']]]

        # Set CPU parallization
        if self.parallel_across == 'folds':
            n_jobs_time = 1
            n_jobs_fold = self.n_jobs
        elif self.parallel_across == 'time_samples':
            n_jobs_time = self.n_jobs
            n_jobs_fold = 1

        # Cross-validation loop
        parallel, p_time_gen, _ = parallel_func(_fit_one_fold, n_jobs_fold)
        packed = parallel(p_time_gen(clone(self.clf), X[train, :,:], y[train],
                          self.train_times['slices'], n_jobs_time)
                          for train, test in self.cv)

        # Unpack in case of parallelization
        unpacked = zip(*packed)

        self.estimators = unpacked

        self.y = y

    def predict(self, epochs, y=None, independent=False, test_times={},
                predict_type='predict', scorer=None, n_jobs=1):
        """ Test each classifier on each specified testing time slice.
        
        Parameters
        ----------
        epochs : object
            MNE epochs. Can be similar to fitted epochs or not. See independent
            parameter.
        y : array | None, optional, default: None
            True model values. If None, y = [epochs.events[:,2]]
        independent : bool
            Indicates whether data X is independent from the data used to fit 
            the  classifier. If independent == True, the predictions from each
            cv fold classifier are averaged. Else, only the prediction from the
            corresponding fold is used.
        test_times : str | dict, optional, default: {} 
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

        test_slices : array, shape(n_train_time, n_test_times), optional, 
            default: None
            Array of time slices (in indices) used to test each classifier.
        test_time_start : float | None, optional, default: None
            Time at which to start testing each classifier (in seconds). By 
            default, min(epochs.times).
        test_time_stop : float | None, optional, default: None
            Maximal time at which to stop testing each classifier (in seconds).
            By default, max(epochs.times).
        test_time_step : float | None, optional, default: None
            Duration separating the start of to subsequent classifiers (in 
            seconds). By default, equals one time sample.
        n_jobs : int
            Number of jobs to run in parallel. Each fold is fit
            in parallel.

        """
        X, y = _format_data(epochs, y, self.chans_picks)

        # Check that at least one classifier has been trained
        assert(len(self.estimators) > 0)

        # Cross validation scheme: if same data set use CV for prediction, else
        # predict each trial with all folds' classifiers
        if not(independent):
            # use cross validation scheme: fit and predict on non-independent
            # trials
            assert(all(y == self.y))
        self.independent = independent

        # Define scorer
        if scorer is None:
            if predict_type == 'predict':
                scorer = accuracy_score
            else:
                scorer = roc_auc_score
        self.scorer = scorer
        self.predict_type = predict_type

        # Define testing sliding window
        if test_times == 'diagonal':
            test_times = {}
            test_times['slices'] = [[s] for s in self.train_times['slices']]
        elif not('slices' in test_times):
            # Initialize array
            test_times['slices_'] = []
            # Force same number of time sample in testing than in training
            # (otherwise it won 't be the same number of features')
            test_times['length'] = self.train_times['length']
            # Make a sliding window for each training time.
            for t in range(0, len(self.train_times['slices'])):
                test_times['slices_'] += [
                    _sliding_window(epochs.times, test_times)]
            test_times['slices'] = test_times['slices_']
            del test_times['slices_']

        # Testing times in milliseconds (only keep last time if multiple time
        # slices)

        test_times['s'] = [[epochs.times[t_test[-1]] for t_test in t_train]
                           for t_train in test_times['slices']]
        # store testing times
        self.test_times = test_times

        # Prepare parallel predictions
        parallel, p_time_gen, _ = parallel_func(_predict_time_loop, n_jobs)

        # Initialize results
        self.y_pred = [[]] * len(test_times['slices'])
        self.scores = [[]] * len(test_times['slices'])
        # Loop across estimators (i.e. training times)
        packed = parallel(p_time_gen(
            X, y,
            self.estimators[t_train],
            self.cv,
            slices,
            self.scorer,
            self.independent, self.predict_type)
            for t_train, slices in enumerate(test_times['slices']))

        self.scores, self.y_pred = zip(*packed)


def _predict_time_loop(X, y, estimator, cv, slices, scorer, independent,
                       predict_type):
    """Aux function of GeneralizationAcrossTime

    Run classifiers predictions loop across time samples.

    Parameters
    ----------
    X : array, shape (n_trials, n_features, n_times)
        To-be-fitted data
    y : list | array, shape (n_trials)
        To-be-fitted model
    estimators : array, shape(n_folds)
        Array of Sklearn classifiers fitted in cross-validation.
    slices : list, shape(n_slices)
        List of slices selecting data from X from which is prediction is 
        generated.
    scorer : object
        Sklearn scoring object
    independent : bool
        Indicates whether data X is independent from the data used to fit the 
        classifier. If independent == True, the predictions from each cv fold
        classifier are averaged. Else, only the prediction from the 
        corresponding fold is used.
    predict_type : str
        Indicates the type of prediction ('predict', 'proba', 'distance').

    Returns
    -------
    scores : array, shape(n_slices)
        Score estimated across all trials for each train/tested time sample.
    y_pred : array, shape(n_slices, n_trials)
        Single trial prediction for each train/tested time sample.
    
    """
    n_trial = len(y)
    n_time = len(slices)
    # Loop across testing slices
    y_pred = [[]] * n_time
    scores = [0] * n_time
    for t, indices in enumerate(slices):
        # Flatten features in case of multiple time samples
        X_train = X[:, :, indices].reshape(
            n_trial, np.prod(X[:, :, indices].shape[1:]))

        # Single trial predictions
        if not(independent):
            # If predict within cross validation, only predict with corresponding
            # classifier, else predict with each fold's classifier and average
            # prediction.
            for k, [train, test] in enumerate(cv):
                # /!\ I didn't manage to initalize correctly this array, as its
                # size depends on the the type of predicter and the number of
                # class.
                if k == 0:
                    y_pred_ = _predicter(X_train[test, :], [estimator[k]], predict_type)
                    y_pred[t] = np.empty((n_trial, y_pred_.shape[1]))
                    y_pred[t][test, :] = y_pred_
                y_pred[t][test, :] = _predicter(X_train[test,:], [estimator[k]], predict_type)
        else:
            y_pred[t] = _predicter(X_train, estimator, predict_type)

        # Scores across trials
        scores[t] = _scorer(y, y_pred[t], scorer)
    return scores, y_pred


def _scorer(y, y_pred, scorer):
    """Aux function of GeneralizationAcrossTime

    Estimate classifiaction score.

    Parameters
    ----------
    y : list | array, shape (n_trials)
        True model value
    y_pred : list | array, shape (n_trials)
        Classifier prediction of model value
    scorer : object
        Sklearn scoring object

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
        # /!\ Problem here with scorer when proba=True but y !=  (0 | 1)
        try:
            score = scorer(y, y_pred)
        except:
            score = scorer(y==max(classes), y_pred)
    else:
        # This part is not sufficiently generic to apply to all classification
        # and regression cases.
        score = 0
        for ii, c in enumerate(classes):
            score += scorer(y == c, y_pred[:, ii])
        score /= len(classes)
    return score


def _format_data(epochs, y, chans_picks):
    """Aux function of GeneralizationAcrossTime

    Format MNE data into Sklearn X and y

    Parameters
    ----------
    epochs : object
        MNE epochs used to train the classifiers (using a cross-
        validation scheme).
    y : array shape(n_trials) | list shape(n_trials) | None
        To-be-fitted model. If y is None, y = epochs.events
    chans_picks : array (n_selected_chans) | None
        Channels to be included in Sklearn model fitting.

    Returns
    -------
    X : array, shape(n_trials, n_selected_chans, n_times)
        To-be-fitted data
    y : array, shape(n_trials)
        To-be-fitted model
    chans_picks : array, shape()
    """
    # If no regressor is passed, use default epochs events
    if y is None:
        y = epochs.events[:, 2]
    # Convert MNE data into trials x features x time matrix
    X = epochs.get_data()[:, chans_picks, :]
    # Check data sets
    assert(X.shape[0] == y.shape[0])
    return X, y


def _fit_one_fold(clf, X, y, train_slices, n_jobs_time):
    """Aux function of GeneralizationAcrossTime

    Run parallelize fitting across time slices.

    Parameters
    ----------
    clf : object
        Sklearn classifier
    X : array, shape (n_trials, n_features, n_times)
        To-be-fitted data
    y : list | array, shape (n_trials)
        To-be-fitted model
    train_slices : array 
        Time samples indices to be included for each classifier
    n_jobs_time : int
        Number of cores to parallelize computation across time slices.

    Returns
    -------
    estimators : array
        array of classifiers for each training time and for each 
        cross-validation fold.
    """
    # Parallel across training time
    parallel, p_time_gen, _ = parallel_func(_fit_one_time, n_jobs_time)
    estimators = parallel(p_time_gen(clone(clf),
                                     X[:, :, train_slices[t_train]], y)
                          for t_train in range(len(train_slices)))
    return estimators


def _fit_one_time(clf, X, y):
    """Aux function of GeneralizationAcrossTime

    Fit each classifier.

    Parameters
    ----------
    clf : object
        Sklearn classifier
    X : array, shape (n_trials, n_features, n_times)
        To-be-fitted data
    y : list | array, shape (n_trials)
        To-be-fitted model

    Returns
    -------
    clf : object
        Sklearn classifier
    """
    # Reshape data matrix to flatten features in case of multiple time samples.
    X_train = X.reshape(len(X), np.prod(X.shape[1:]))
    # Fit classifier
    clf.fit(X_train, y)
    return clf


def _sliding_window(times, options):
    """Aux function of GeneralizationAcrossTime

    Define the slices on which to train each classifier.

    Parameters
    ----------
    times : array, shape (n_times)
        Array of times from MNE epochs
    options : dict, optional keys: ('start', 'stop', 'step', 'length' )
        'start' : float
            Minimum time at which to stop decoding (in seconds). By default, 
            max(times).
        'stop' : float
            Maximal time at which to stop decoding (in seconds). By default, 
            max(times).
        'step' : float
            Duration separating the start of to subsequent classifiers (in 
            seconds). By default, equals one time sample.
        'length' : float
            Duration of each classifier (in seconds). By default, equals one time sample.

    Returns
    -------
    time_pick : list, shape(n_classifiers)
        List of training slices, indicating for each classifier the time sample 
        (in indices of times) to be fitted on.
    """

    # Sampling frequency
    freq = (times[-1] - times[0]) / len(times)

    # Default values
    if ('slices' in options) and np.all([key in options
        for key in ('start', 'stop', 'step', 'length')]):
        time_pick = options['slices']
    else:
        if not('start' in options):
            options['start'] = times[0]
        if not('stop' in options):
            options['stop'] = times[-1]
        if not('step' in options):
            options['step'] = freq
        if not('length' in options):
            options['length'] = freq

        # Convert seconds to index

        def find_time(t):
            if any(times >= t):
                return np.nonzero(times >= t)[0][0]
            else:
                print('Timing outside limits!')
                raise

        start = find_time(options['start'])
        stop = find_time(options['stop'])
        step = int(round(options['step'] / freq))
        length = int(round(options['length'] / freq))

        # For each training slice, give time samples to be included
        time_pick = [range(start, start + length)]
        while (time_pick[-1][0] + step) <= (stop - length + 1):
            start = time_pick[-1][0] + step
            time_pick += [range(start, start + length)]

    return time_pick


def _predicter(X, estimators, predict_type):
    """Aux function of GeneralizationAcrossTime

    Predict each classifier. If multiple classifiers are passed, average 
    prediction across all classifier to result in a single prediction per 
    classifier.

    Parameters
    ----------
    estimators : array, shape(n_folds) or shape(1)
        Array of Sklearn classifiers to predict data
    X : array, shape (n_trials, n_features, n_times)
        To-be-predicted data
    predict_type : str, 'predict' | 'distance' | 'proba'
        'predict' : simple prediction of y (e.g. SVC, SVR)
        'distance': continuous prediction (e.g. decision_function)
        'proba': probabilistic prediction (e.g. SVC(probability=True))

    Returns
    -------
    y_pred : array, shape(n_trials, m_prediction_dimensions)
        Classifier's prediction for each trial.
    """
    # Initialize results:
    # /!\ Here I did not manage to find an efficient and generic way to guess
    # the number of output provided by predict, and could thus not initalize
    # the y_pred values.
    n_trial = X.shape[0]
    n_clf = len(estimators)
    if predict_type == 'predict':
        n_class = 1
    elif predict_type == 'distance':
        n_class = estimators[0].decision_function(X[0, :]).shape[1]
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
            y_pred[:, :, fold] = clf.decision_function(X)

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


def plot_time_gen(gat, title=None, vmin=0., vmax=1., tlim=None, ax=None,
                  show=True):
    """Plotting function of GeneralizationAcrossTime object

    Predict each classifier. If multiple classifiers are passed, average 
    prediction across all classifier to result in a single prediction per 
    classifier.

    Parameters
    ----------
    gat : object
        GeneralizationAcrossTime object containing predictions.
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
    show : bool, optional, default: True
        plt.show()       

    Returns
    -------
    ax : object
        Plot pointer.
    """

    # Check that same amount of testing time per training time
    assert(len(np.unique([len(t) for t in gat.test_times])))
    # Setup plot
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # Define time limits
    if tlim is None:
        tlim = [gat.train_times['s'][0], gat.train_times['s'][-1],
                gat.test_times['s'][0][0], gat.test_times['s'][-1][-1]]
    # Plot scores
    im = ax.imshow(gat.scores, interpolation='nearest', origin='lower',
                   extent=tlim, vmin=vmin, vmax=vmax)
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    if not(title is None):
        ax.set_title(title)
    ax.axvline(0, color='k')
    ax.axhline(0, color='k')
    plt.colorbar(im, ax=ax)
    if show:
        plt.show()
    return im, ax


def plot_decod(gat, title=None, ymin=0., ymax=1., ax=None, show=True, color='b'):
    """Plotting function of GeneralizationAcrossTime object

    Predict each classifier. If multiple classifiers are passed, average 
    prediction across all classifier to result in a single prediction per 
    classifier.

    Parameters
    ----------
    gat : object
        GeneralizationAcrossTime object containing predictions.
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
    ax : object
        Plot pointer.
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    # detect whether gat is a full matrix or just its diagonal
    if np.all(np.unique([len(t) for t in gat.test_times['s']]) == 1):
        scores = gat.scores
    else:
        scores = np.diag(gat.scores)
    ax.plot(gat.train_times['s'], scores, color=color, label="Classif. score")
    ax.axhline(0.5, color='k', linestyle='--', label="Chance level")
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(gat.scorer.func_name)
    ax.legend(loc='best')
    if show:
        plt.show()
    return ax
