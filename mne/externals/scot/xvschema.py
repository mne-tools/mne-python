# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

""" Cross-validation schemas """

from numpy import sort


def singletrial(t, num_trials):
    """ Single-trial cross-validation schema
    
    Use one trial for training, all others for testing.

    Parameters
    ----------
    t : int
        Current cross-validation run
    num_trials : int
        Total number of trials
        
    Returns
    -------
    trainset : list of int
        trials in the training set
    testset : list of int
        trials in the testing set
    """
    trainset = [t]
    testset = [i for i in range(trainset[0])] + [i for i in range(trainset[-1] + 1, num_trials)]

    testset = sort([t % num_trials for t in testset])

    return trainset, testset


def multitrial(t, num_trials):
    """ Single-trial cross-validation schema
    
    Use one trial for testing, all others for training.

    Parameters
    ----------
    t : int
        Current cross-validation run
    num_trials : int
        Total number of trials
        
    Returns
    -------
    trainset : list of int
        trials in the training set
    testset : list of int
        trials in the testing set
    """
    testset = [t]
    trainset = [i for i in range(testset[0])] + [i for i in range(testset[-1] + 1, num_trials)]

    trainset = sort([t % num_trials for t in trainset])

    return trainset, testset