# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from ..utils import logger, verbose
from ..parallel import parallel_func
from ..pick import channel_type, pick_types


def _time_gen_one_fold(clf, scorer, 
                       X, y, 
                       X_generalize, y_generalize, 
                       train, test, 
                       train_times, test_times):
    """Aux function of time_generalization"""
    
    scores = np.zeros((len(train_times), len(test_times)))
    generalize_across_condition = X_generalize != None and y_generalize != None
    if generalize_across_condition:
        scores_generalize = np.zeros((len(train_times), len(test_times)))
    else:
        scores_generalize = None
    
    # loops across time points
    my_reshape = lambda X: X.reshape(len(X), np.prod(X.shape[1:]))
    for train_time, t_train in enumerate(train_times):
        # select time slide
        X_train = my_reshape(X[train, :, train_time])
        clf.fit(X_train, y[train])
        for test_time, t_test in enumerate(test_times):
            X_test = my_reshape(X[test, :, test_time])
            scores[t_test, t_train] = scorer(clf, X_test, y[test])
            # Generalize across experimental conditions?
            if generalize_across_condition:
                x_gen = my_reshape(X_generalize[:, :, test_time])
                scores_generalize[t_test, t_train] = scorer(clf, x_gen, y_generalize)
    return scores, scores_generalize


def create_slices(n_times, start=0, stop=None, width=1, across_step=None, within_step=1):
    """ Generate slices of time 
    Parameters
    ----------
    n_times : integer
        number of total time samples
    start : integer
        index where first slice should start
    stop : integer
        index where last slice should maximally end
    width : integer
        number of time sample included in a given slice
    across_step: integer
        number of time samples separating two slices
    within_step: integer
        number of time samples separating two temporal feature within a slice of time

    Returns
    -------
    slices : list 
        list of list of time indexes

    Notes
    ----------
    This function may be changed to become more general and fit frequency and spatial slicing (i.e. search light)"""
    
    # default parameters
    if stop is None: stop = n_times
    if across_step is None: across_step = width 
    
    # slicing
    slices = [slice(t,t+width,within_step) for t in
            range(start, stop - width + 1, across_step)]
    return slices


@verbose
def time_generalization(epochs_list, epochs_list_generalize=None, clf=None, cv=5, scoring="roc_auc",
                        window_width=1, shuffle=True, random_state=None, n_jobs=1,
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
        These epcohs are used to train the classifiers (using a cross-validation scheme)
    epochs_list_generalize : list of Epochs
        Epochs used to test the classifiers' generalization performance 
        in novel experimental conditions.
    window_width : integer
        Number of time samples considered by each classifier
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

    # Apply same procedure with optional generalization set
    if epochs_list_generalize == None:
        X_generalize = None
        y_generalize = None
    else:
        info = epochs_list_generalize[0].info
        data_picks = pick_types(info, meg=True, eeg=True, exclude='bads')
        X_generalize = [e.get_data()[:, data_picks, :] for e in epochs_list_generalize]
        y_generalize = [k * np.ones(len(this_X)) for k, this_X in enumerate(X_generalize)]
        X_generalize = np.concatenate(X_generalize)
        y_generalize = np.concatenate(y_generalize)

    # Launch main script
    ch_types = set([channel_type(info, idx) for idx in data_picks])
    logger.info('Running time generalization on %s epochs using %s.' %
                (len(X), ch_types.pop()))

    out = time_gen_all_fold(X,y,clf=clf, scoring=scoring, cv=cv, n_jobs=n_jobs,
        window_width=window_width,X_generalize=X_generalize, y_generalize=y_generalize)

    return out



def time_gen_all_fold(X, y, clf=None, scoring="roc_auc",
    shuffle=True, random_state=None, cv=5, n_jobs=1,
    window_width=1, X_generalize=None, y_generalize=None):
    """ This functions allows users using the pipeline direclty with X and y, rather than MNE structured data """
    from sklearn.cross_validation import check_cv
    from sklearn.base import clone
    from sklearn.utils import check_random_state
    from sklearn.metrics import SCORERS
    from nose.tools import assert_true

    # check data sets
    assert_true(X.shape[0] == y.shape[0])
    if X_generalize is not None and y_generalize is not None:
        assert_true(X_generalize.shape[0] == y_generalize.shape[0])

    # re-order data to avoid taking to avoid folding bias
    if shuffle:
        rng = check_random_state(random_state)
        order = np.argsort(rng.randn(len(X)))
        X = X[order]
        y = y[order]

    # Set default MVPA: support vector classifier
    if clf is None:
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        svc = SVC(C=1, kernel='linear')
        clf = Pipeline([('scaler', scaler), ('svc', svc)])

    # Set default cross validation scheme
    cv = check_cv(cv, X, y, classifier=True)

    # Set default scoring scheme
    scorer = SCORERS[scoring]

    # setup temporal generalization slicing
    n_times = X.shape[2]-window_width+1
    train_times = create_slices(n_times, width=window_width)
    test_times = create_slices(n_times, width=window_width)
    
    # Run parallel decoding across folds
    parallel, p_time_gen, _ = parallel_func(_time_gen_one_fold, n_jobs)
    scores = parallel(p_time_gen(clone(clf), scorer,
        X, y, 
        X_generalize, y_generalize, 
        train, test, 
        train_times, test_times)
                      for train, test in cv)

    # Unpack MVPA results & output results as a dictionary
    scores, scores_generalize = zip(*scores)
    scores = np.mean(scores, axis=0)
    out = dict(scores=scores)
    if X_generalize is not None:
        scores_generalize = np.mean(scores_generalize, axis=0)
        out['scores_generalize'] = scores_generalize
    out['train_times'] = train_times
    out['test_times'] = train_times
    return out

# to do list
# - ouput classifiers rescaled coefficients
# - pass more window slicing arguments to user
# - design different window parameters tests
# - generalize slicing function, put in utils?

