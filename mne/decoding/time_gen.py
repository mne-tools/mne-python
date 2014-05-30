# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from ..utils import logger, verbose, create_slices
from ..parallel import parallel_func
from ..pick import channel_type, pick_types


def _time_gen_one_fold(clf, scorer, 
                       X, y, 
                       X_generalize, y_generalize, 
                       train, test, 
                       train_times, test_times,
                       compress_results=True):
    """Aux function of time_generalization"""
    
    # Initialize results
    n_train_t = max([t.stop for t in train_times]) # get maximum time sample
    n_test_t = max([t.stop for tt in test_times for t in tt])
    scores = np.zeros((n_train_t, n_test_t))
    tested = np.zeros((n_train_t, n_test_t), dtype=bool)
    generalize_across_condition = X_generalize is not None and y_generalize is not None
    if generalize_across_condition:
        scores_generalize = np.zeros((n_train_t, n_test_t))
    else:
        scores_generalize = None
    
    # Function to vectorize all time * channels features
    my_reshape = lambda X: X.reshape(len(X), np.prod(X.shape[1:]))
    # Loop across time points
    for t_train, train_time in enumerate(train_times):
        # Select training time slice
        X_train = my_reshape(X[train, :, train_time])
        clf.fit(X_train, y[train])
        for test_time in test_times[t_train]:
            # Select testing time slice
            X_test = my_reshape(X[test, :, test_time])
            # Evaluate classifer on cross-validation set
            scores[train_time.start, test_time.start] = scorer(clf, X_test, y[test])
            tested[train_time.start, test_time.start] = True
            # Evaluate classifier on cross-condition generalization set
            if generalize_across_condition:
                x_gen = my_reshape(X_generalize[:, :, test_time])
                scores_generalize[train_time.start, test_time.start] = scorer(clf, x_gen, y_generalize)

    if compress_results:
        # avoid returning partially empty results
        # removing empty lines and columns (generally due to window width > 1)
        scores = scores[:,np.any(tested,axis=1)]
        scores = scores[np.any(tested, axis=0),:]
        if generalize_across_condition:
            scores_generalize = scores_generalize[:,np.any(tested,axis=1)]
            scores_generalize = scores_generalize[np.any(tested,axis=0),:]
    return scores, scores_generalize

@verbose
def time_generalization(epochs_list, epochs_list_generalize=None, 
                        clf=None, cv=5, scoring="roc_auc",
                        train_times=None, test_times=None,
                        shuffle=True, random_state=None, 
                        n_jobs=1, verbose=None):
    """Fit decoder at each time instant and test at all others

    The function returns the cross-validation scores when the train set
    is from one time instant and the test from all others.

    The decoding will be done using all available data channels, but
    will only work if 1 type of channel is availalble. For example
    epochs should contain only gradiometers.

    Parameters
    ----------
    epochs_list : list
        These epcohs are used to train the classifiers (using a cross-validation 
        scheme).
    epochs_list_generalize : list | None
        Epochs used to test the classifiers' generalization performance 
        in novel experimental conditions.
    train_times : list | callable | None
        List of slices generated with create_slices()
    test_times : list |  callable | None
        List of slices generated with create_slices()
    clf : object | None
        A object following scikit-learn estimator API (fit & predict).
        If None the classifier will be a linear SVM (C=1.) after
        feature standardization.
    cv : integer | object
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
    scores : array, shape (training_times, testing_times)
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
    if epochs_list_generalize is None:
        X_generalize = None
        y_generalize = None
    else:
        info = epochs_list_generalize[0].info
        data_picks = pick_types(info, meg=True, eeg=True, exclude='bads')
        X_generalize = [e.get_data()[:, data_picks, :] 
                        for e in epochs_list_generalize]
        y_generalize = [k * np.ones(len(this_X)) 
                        for k, this_X in enumerate(X_generalize)]
        X_generalize = np.concatenate(X_generalize)
        y_generalize = np.concatenate(y_generalize)

    # Launch main script
    ch_types = set([channel_type(info, idx) for idx in data_picks])
    logger.info('Running time generalization on %s epochs using %s.' %
                (len(X), ch_types.pop()))

    out = time_generalization_Xy(X, y, 
                                 X_generalize=X_generalize, 
                                 y_generalize=y_generalize,
                                 clf=clf, scoring=scoring, cv=cv, 
                                 train_times=train_times,
                                 test_times=test_times, 
                                 n_jobs=n_jobs)

    return out



def time_generalization_Xy(X, y, X_generalize=None, y_generalize=None,
                           clf=None, scoring="roc_auc", cv=5, 
                           train_times=None, test_times=None,
                           shuffle=True, random_state=None,
                           n_jobs=1):
    """ This functions allows users using the pipeline direclty with X and y, 
    rather than MNE  structured data 

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_times)
        Input data on which the model is fitted (with cross-validation).
    y : array, shape (n_trials)
        To-be-fitted model (e.g. trials' classes).
    X_generalize : array, shape (m_trials, n_channels, n_times) | None
        Input data on which the model ability to generalize to a novel condition 
        is tested.
    y_generalize : array, shape (m_trials) | None
        Generalization model.
    
    The other input parameters are identical to time_generalization().
    
    Returns
    -------
    out : dict
        'scores' : mean cross-validated scores across folds
        'scores_generalize' : mean cross-condition generalization scores
        'time_train' : time slices used to train each classifier
        'time_test' : time slices used to test each classifier
    """
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

    # Setup temporal generalization slicing
    if train_times is None:
        # default: train and test over all time samples
        train_times = create_slices(X.shape[2]) 
    elif callable(train_times):
        # create slices once n_times is known
        train_times = train_times(X.shape[2]) 
    
    if test_times is None: 
        # default: testing time is identical to training time
        test_times = [train_times]*X.shape[2]
    elif callable(test_times):
        # create slices once n_times is known
        test_times = [test_times(X.shape[2])]*X.shape[2]
    
    # Run parallel decoding across folds
    parallel, p_time_gen, _ = parallel_func(_time_gen_one_fold, n_jobs)
    scores = parallel(p_time_gen(clone(clf), scorer, X, y, X_generalize,
                      y_generalize, train, test, train_times, test_times)
                      for train, test in cv)

    # Unpack MVPA results from parallel outputs
    scores, scores_generalize = zip(*scores)
    scores = np.mean(scores, axis=0)

    # Output results in a dictionary to allow future extensions
    out = dict(scores=scores)
    if X_generalize is not None:
        scores_generalize = np.mean(scores_generalize, axis=0)
        out['scores_generalize'] = scores_generalize

    out['train_times'] = train_times
    out['test_times'] = test_times
    
    return out
