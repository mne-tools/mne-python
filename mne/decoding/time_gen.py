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
                       X_gen, y_gen, 
                       train, test, 
                       train_slices, test_slices,
                       compress_results=True):
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
    compress_results : bool
        If True, removes time dimensions neither used for clf testing.
    """
    
    # Initialize results
    n_train_t = max([t.stop for t in train_slices]) # get maximum time sample
    n_test_t = max([t.stop for tt in test_slices for t in tt])
    scores = np.zeros((n_train_t, n_test_t)) # scores
    tested = np.zeros((n_train_t, n_test_t), dtype=bool) # tested time points
    generalize_across_condition = X_gen is not None and y_gen is not None
    if generalize_across_condition:
        scores_gen = np.zeros((n_train_t, n_test_t))
    else:
        scores_gen = None
    
    # Function to vectorize all time * channels features
    my_reshape = lambda X: X.reshape(len(X), np.prod(X.shape[1:]))
    # Loop across time points
    for t_train, train_time in enumerate(train_slices):
        # Select training time slice
        X_train = my_reshape(X[train, :, train_time])
        clf.fit(X_train, y[train])
        for test_time in test_slices[t_train]:
            # Select testing time slice
            X_test = my_reshape(X[test, :, test_time])
            # Evaluate classifer on cross-validation set
            scores[train_time.start, test_time.start] = scorer(clf, X_test, y[test])
            tested[train_time.start, test_time.start] = True
            # Evaluate classifier on cross-condition generalization set
            if generalize_across_condition:
                x_gen = my_reshape(X_gen[:, :, test_time])
                scores_gen[train_time.start, test_time.start] = scorer(clf, x_gen, y_gen)

    if compress_results:
        # avoid returning partially empty results
        # removing empty lines and columns (generally due to window width > 1)
        scores = scores[:,np.any(tested, axis=0)]
        scores = scores[np.any(tested, axis=1),:]
        if generalize_across_condition:
            scores_gen = scores_gen[:,np.any(tested,axis=0)]
            scores_gen = scores_gen[np.any(tested,axis=1),:]
    return scores, scores_gen

@verbose
def time_generalization(epochs_list, epochs_list_gen=None, 
                        clf=None, cv=5, scoring="roc_auc",
                        generalization="cardinal",
                        train_slices=None, test_slices=None,
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
    epochs_list_gen : list | None
        Epochs used to test the classifiers' generalization performance 
        in novel experimental conditions.
    train_slices : list | callable | None
        List of slices generated with create_slices()
    test_slices : list |  callable | None
        List of slices generated with create_slices()
    generalization: str
        "cardinal" or "diagonal" to construct relative or absolute testing 
        slices from train slices
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
    scores : array, shape (training_slices, testing_slices)
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
    if epochs_list_gen is None:
        X_gen = None
        y_gen = None
    else:
        info = epochs_list_gen[0].info
        data_picks = pick_types(info, meg=True, eeg=True, exclude='bads')
        X_gen = [e.get_data()[:, data_picks, :] 
                        for e in epochs_list_gen]
        y_gen = [k * np.ones(len(this_X)) 
                        for k, this_X in enumerate(X_gen)]
        X_gen = np.concatenate(X_gen)
        y_gen = np.concatenate(y_gen)

    # Setup time slices
    n_sample = X.shape[2]
    # Change code here to add timing (ms -> sample) compatibility 
    train_slices, test_slices = gen_type(n_sample,
                                         generalization=generalization, 
                                         train_slices=train_slices,
                                         test_slices=test_slices)
    # Launch main script
    ch_types = set([channel_type(info, idx) for idx in data_picks])
    logger.info('Running time generalization on %s epochs using %s.' %
                (len(X), ch_types.pop()))

    out = time_generalization_Xy(X, y, 
                                 X_gen=X_gen, 
                                 y_gen=y_gen,
                                 clf=clf, scoring=scoring, cv=cv, 
                                 train_slices=train_slices,
                                 test_slices=test_slices, 
                                 n_jobs=n_jobs)

    out['train_times'] = epochs_list[0].times[[s.start for s in out['train_slices']]]
    out['test_times'] = epochs_list[0].times[[s.start for s in out['test_slices'][0]]]

    return out


def gen_type(n_sample, generalization='diagonal', train_slices=None, 
             test_slices=None):
    """ Creates typical temporal generalization scenarios

    The function return train_slices, test_slices that indicate the time 
    samples to be used for training and testing each classifier. These
    lists can be directly used by time_generalization_Xy()

    Parameters
    ----------
    n_sample : int
        Number of time samples in each trial | Last sample to on which the
        classifier can be trained
    generalization : str, value = 'diagonal' | 'cardinal'
        Indicates the type of scenario used for the testing_slices
    train_slices : list | object | None
        create_slices() partial object or list of training slices
    train_slices : list | object | None
        create_slices() partial object or list of test slices
    """
    
    # Setup train slices
    if train_slices is None:
        # default: train and test over all time samples
        train_slices = create_slices(n_sample) 
    elif callable(train_slices):
        # create slices once n_slices is known
       train_slices = train_slices(n_sample) 
    
    # Setup test slices
    if generalization =='cardinal':
        # Time generalization is from/to particular time samples
        if test_slices is None: 
            # Default: testing time is identical to training time
            test_slices = [train_slices] * len(train_slices)
        elif callable(test_slices):
            test_slices = [test_slices(n_sample)] * len(train_slices)

    elif generalization == 'diagonal':
        # Time generalization is at/around the training time samples
        if test_slices is None: 
            # Default: testing times are identical to training slices 
            # (classic decoding across time)
            test_slices = [[s] for s in train_slices]
        else:
            def up_slice(test,train):
                """ Update slice by combining timing of test and train slices 
                """
                out = slice(test.start + train.start, 
                            test.stop + train.stop - 1, 
                            train.step)
                return out
            test_slices = np.tile([test_slices], (len(train_slices),1)).tolist()
            for t_train in range(len(train_slices)):
                for t_test in range(len(test_slices[t_train])):
                    # Add start and stop of training and testing slices
                    # to make testing timing dependent on training timing
                    test_slices[t_train][t_test] = up_slice(test_slices[t_train][t_test],
                                                            train_slices[t_train])

    # Check that all time samples are in bounds
    if any([s.start < 0 | s.stop > n_sample for s in train_slices]) | \
       any([s.start < 0 | s.stop > n_sample for ss in test_slices for s in ss]):
        logger.info('/!\ Slicing: time samples out of bound!')
        
        # Shortcut to select slices that are in bounds
        sel = lambda slices, bol: [s for (s, b) in zip(slices, bol) if b]
        
        # Deal with testing slices first:
        for t_train in range(len(test_slices)):
            # Find testing slices that are in bounds
            inbound = [(s.start >= 0) & (s.stop <= n_sample) 
                       for s in test_slices[t_train]]
            test_slices[t_train] = sel(test_slices[t_train],inbound)

        # Deal with training slices then:
        inbound = [(s.start >= 0) & (s.stop <= n_sample) for s in train_slices]
        train_slices = sel(train_slices,inbound)

    return train_slices, test_slices



def time_generalization_Xy(X, y, X_gen=None, y_gen=None,
                           clf=None, scoring="roc_auc", cv=5, 
                           train_slices=None, test_slices=None,
                           shuffle=True, random_state=None,
                           compress_results=True, n_jobs=1):
    """ This functions allows users using the pipeline direclty with X and y, 
    rather than MNE  structured data 

    Parameters
    ----------
    X : array, shape (n_trials, n_channels, n_slices)
        Input data on which the model is fitted (with cross-validation).
    y : array, shape (n_trials)
        To-be-fitted model (e.g. trials' classes).
    X_gen : array, shape (m_trials, n_channels, n_slices) | None
        Input data on which the model ability to generalize to a novel condition 
        is tested.
    y_gen : array, shape (m_trials) | None
        Generalization model.
    
    The other input parameters are identical to time_generalization().
    
    Returns
    -------
    out : dict
        'scores' : mean cross-validated scores across folds
        'scores_gen' : mean cross-condition generalization scores
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
    if X_gen is not None and y_gen is not None:
        assert_true(X_gen.shape[0] == y_gen.shape[0])

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
    if train_slices is None:
        # Default: train and test over all time samples
        train_slices = create_slices(X.shape[2]) 
    elif callable(train_slices):
        # Create slices once n_slices is known
        train_slices = train_slices(X.shape[2]) 

    if test_slices is None: 
        # Default: testing time is identical to training time
        test_slices = [train_slices] * len(train_slices)
    elif callable(test_slices):
        # Create slices once n_slices is known
        test_slices = [test_slices(X.shape[2])] * len(train_slices)
    
    # Run parallel decoding across folds
    parallel, p_time_gen, _ = parallel_func(_time_gen_one_fold, n_jobs)
    scores = parallel(p_time_gen(clone(clf), scorer, X, y, X_gen,
                      y_gen, train, test, train_slices, test_slices)
                      for train, test in cv)

    # Unpack MVPA results from parallel outputs
    scores, scores_gen = zip(*scores)
    scores = np.mean(scores, axis=0)

    # Output results in a dictionary to allow future extensions
    out = dict(scores=scores)
    if X_gen is not None:
        scores_gen = np.mean(scores_gen, axis=0)
        out['scores_gen'] = scores_gen

    out['train_slices'] = train_slices
    out['test_slices'] = test_slices
    
    return out
