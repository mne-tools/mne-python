# Authors: Romain Trachel <trachelr@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np

class LinearClassifier():
    """
    This object clones a Linear Classifier from sklearn
    and updates the attribute for each fit. The model coefficient
    can be interpreted using the attribute patterns [1].

    Parameters
    ----------
    clf : object | None
        A linear classifier from sklearn with a fit method 
        that updates a coef_ attribute.
        If None the classifier will be a LogisticRegression
    info : dict
        measurement info

    Attributes
    ----------
    filters_ : ndarray
        If fit, the filters used to decompose the data, else None.
    patterns_ : ndarray
        If fit, the patterns used to restore M/EEG signals, else None.
    
    References
    ----------
    [1] Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J.-D., 
    Blankertz, B., & Bießmann, F. (2014). On the interpretation of 
    weight vectors of linear models in multivariate neuroimaging. 
    NeuroImage, 87, 96–110. doi:10.1016/j.neuroimage.2013.10.067
    """
    def __init__(self, clf, info):
        self.clf = clf
        self.info = info
    
    
    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)."
                             % type(X))
        # check for features dimension
        X = np.atleast_2d(X)
        if len(X.shape) != 2:
            raise ValueError("X dimension should be 2 (n_trials x n_features)"
                             " instead of ", X.shape)
        # check for number of classes
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError("Need at least two different classes in the data.")
        
        # fit the classifier
        clf.fit(X, y)
        # computes the patterns
        if hasattr(clf, 'coef_'):
            self.patterns_ = np.dot(X.T, np.dot(X, clf.coef_.T))
            self.filters_ = clf.coef_
    
        
        