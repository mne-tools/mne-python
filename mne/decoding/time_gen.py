# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np

from ..utils import logger, verbose
from ..parallel import parallel_func
from ..io.pick import channel_type, pick_types


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

    ch_types = set([channel_type(info, idx) for idx in data_picks])
    logger.info('Running time generalization on %s epochs using %s.' %
                (len(X), ch_types.pop()))

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
