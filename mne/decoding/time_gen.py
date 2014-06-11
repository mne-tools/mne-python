# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

from ..utils import logger, verbose, create_slices
from ..parallel import parallel_func
from ..pick import channel_type, pick_types


def _one_fold(clf, scorer, X, y, X_gen, y_gen, train, test, train_slices,
              test_slices, n_jobs=1):
    """Aux function of time_generalization

    Parameters
    ----------
    clf : object
        Sklearn classifier
    scorer : object
        Sklearn object
    X : array, shape (n_trials, n_features, n_times)
        To-be-fitted data
    y : list | array, shape (n_trials)
        To-be-fitted model
    X_gen : array, shape (m_trials, n_features, n_times)
        Data used solely for clf testing
    y_gen : list | array, shape (m_trials)
        Model used solely for clf testing

    Returns
    -------
    scores : array
        Classification scores at each training/testing sample.
    scores_gen : array
        Classification scores of generalization set at each training/testing
        sample.
    tested : bool array
        Indicate which training/testing sample was used.
    """

    from sklearn.base import clone

    # Initialize results
    n_train_t = max([t.stop for t in train_slices])  # get maximum time sample
    n_test_t = max([t.stop for tt in test_slices for t in tt])
    scores = np.zeros((n_train_t, n_test_t))  # scores
    tested = np.zeros((n_train_t, n_test_t), dtype=bool)  # tested time points
    if (X_gen is not None) and (y_gen is not None):
        scores_gen = np.zeros((n_train_t, n_test_t))
    else:
        scores_gen = None

    # Loop across time points
    # Parallel across training time
    parallel, p_time_gen, _ = parallel_func(_time_loop, n_jobs)
    packed = parallel(p_time_gen(clone(clf), scorer, X, y, train, test,
                                 X_gen, y_gen, train_slices[t_train],
                                 test_slices[t_train])
                      for t_train in range(len(train_slices)))
    # Unpack results in temporary variables
    scores_, scores_gen_, tested_ = zip(*packed)

    # Store results in absolute sampling-time
    for t_train, train_time in enumerate(train_slices):
        for t_test, test_time in enumerate(test_slices[t_train]):
            scores[train_time.start, test_time.start] = scores_[
                t_train][t_test]
            tested[train_time.start, test_time.start] = tested_[
                t_train][t_test]
            if (X_gen is not None) and (y_gen is not None):
                scores_gen[train_time.start, test_time.start] = scores_gen_[
                    t_train][t_test]
    return scores, scores_gen, tested


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
        if (X_gen is not None) and (y_gen is not None):
            x_gen = my_reshape(X_gen[:, :, test_slice])
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
        logger.info('/!\ Slicing: time samples out of bound!')

        # Shortcut to select slices that are in bounds
        sel = lambda slices, bol: [s for (s, b) in zip(slices, bol) if b]

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


@verbose
def time_generalization(epochs_list, epochs_list_gen=None, clf=None,
                        scoring="roc_auc", cv=5, train_slices=None,
                        test_slices=None, relative_test_slice=False,
                        shuffle=True, random_state=None,
                        compress_results=True, n_jobs=1,
                        parallel_across='folds', verbose=None):
    """Fit decoder at each time instant and test at all others

    The function returns the cross-validation scores when the train set
    is from one time instant and the test from all others.

    The decoding will be done using all available data channels, but
    will only work if 1 type of channel is availalble. For example
    epochs should contain only gradiometers.

    Parameters
    ----------
    epochs_list : list
        These epochs are used to train the classifiers (using a cross-
        validation scheme).
    epochs_list_gen : list | None
        Epochs used to test the classifiers' generalization performance
        in novel experimental conditions.
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
    train_slices : list | callable | None
        List of slices generated with create_slices(). By default the
        classifiers are trained on all time points (i.e.
        create_slices(n_time)).
    test_slices : list |  callable | None
        List of slices generated with create_slices(). By default the
        classifiers are tested on all time points (i.e.
        [create_slices(n_time)] * n_time).
    relative_test_slice: bool
        True implies that the samples indicated in test_slices are relative to
        the samples in train_slices. False implies that the samples in 
        test_slices corresponds to the actual data samples.
    compress_results : bool
        If true returns only training/tested time samples.
    n_jobs : int
        Number of jobs to run in parallel. Each fold is fit
        in parallel.
    parallel_across : str, 'folds' | 'time_samples'
        Set the parallel (multi-core) computation across folds or across
        time samples.

    Returns
    -------
    out : dict
        'scores' : array, shape (training_slices, testing_slices)
                   The cross-validated scores averaged across folds. 
                   scores[i, j] contains  the generalization score when 
                   learning at time j and testing at time i. The diagonal
                   is the cross-validation score at each time-independant 
                   instant.
        'scores_gen' : array, shape (training_slices, testing_slices)
                       identical to scores for cross-condition generalization
                       (i.e. epochs_list_gen)
        'train_times' : first time samples used to train each classifier
        'train_times' : first time samples used to test each classifier

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
    from sklearn.metrics import SCORERS

    # Extract MNE data
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
    n_trials, n_channels, n_samples = X.shape

    # Apply same procedure with optional generalization set
    if epochs_list_gen is None:
        X_gen, y_gen = None, None
    else:
        info = epochs_list_gen[0].info
        data_picks = pick_types(info, meg=True, eeg=True, exclude='bads')
        X_gen = [e.get_data()[:, data_picks, :]
                 for e in epochs_list_gen]
        y_gen = [k * np.ones(len(this_X)) for k, this_X in enumerate(X_gen)]
        X_gen = np.concatenate(X_gen)
        y_gen = np.concatenate(y_gen)

    # check data sets
    assert(X.shape[0] == y.shape[0] == n_trials)
    if X_gen is not None and y_gen is not None:
        assert(X_gen.shape[0] == y_gen.shape[0])

    # re-order data to avoid taking to avoid folding bias
    if shuffle:
        rng = check_random_state(random_state)
        order = np.argsort(rng.randn(n_trials))
        X = X[order]
        y = y[order]

    # Set default MVPA: support vector classifier
    if clf is None:
        scaler = StandardScaler()
        svc = SVC(C=1, kernel='linear')
        clf = Pipeline([('scaler', scaler), ('svc', svc)])

    # Set default cross validation scheme
    cv = check_cv(cv, X, y, classifier=True)

    # Set default scoring scheme
    if type(scoring) is str:
        scorer = SCORERS[scoring]
    else:
        scorer = scoring

    # Set default train and test slices
    train_slices, test_slices = _gen_type(n_samples,
                                         relative_test_slice=relative_test_slice,
                                         train_slices=train_slices,
                                         test_slices=test_slices)

    # Chose parallization type
    if parallel_across == 'folds':
        n_jobs_time = 1
        n_jobs_fold = n_jobs
    elif parallel_across == 'time_samples':
        n_jobs_time = n_jobs
        n_jobs_fold = 1

    # Launch main script
    ch_types = set([channel_type(info, idx) for idx in data_picks])
    logger.info('Running time generalization on %s epochs using %s.' %
                (len(X), ch_types.pop()))

    # Cross-validation loop
    parallel, p_time_gen, _ = parallel_func(_one_fold, n_jobs_fold)
    packed = parallel(p_time_gen(clone(clf), scorer, X, y, X_gen,
                                 y_gen, train, test, train_slices, test_slices,
                                 n_jobs=n_jobs_time)
                      for train, test in cv)

    # Unpack MVPA results from parallel outputs
    scores, scores_gen, tested = zip(*packed)

    # Mean scores across folds
    scores = np.mean(scores, axis=0)
    tested = tested[0]

    # Simplify results
    if compress_results:
        scores = _compress_results(scores, tested)

    # Output results in a dictionary to allow future extensions
    out = dict(scores=scores)
    if X_gen is not None:
        scores_gen = np.mean(scores_gen, axis=0)
        if compress_results:
            scores_gen = _compress_results(scores_gen, tested)
        out['scores_gen'] = scores_gen

    out['train_times'] = epochs_list[0].times[
        [s.start for s in train_slices]]
    out['test_times'] = epochs_list[0].times[
        [s.start for s in test_slices[0]]]

    return out
