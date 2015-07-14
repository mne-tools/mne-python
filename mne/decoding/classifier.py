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
    """
    def __init__(self, clf):
        self.clf = clf
    
    def fit(self, X, y):
        """Estimate the coeffiscient of the linear classifier.
        Save the coeffiscient in the attribute filters_ and 
        computes the attribute patterns_ using [1].

        Parameters
        ----------
        X : array, shape=(n_trials, n_features)
            The data to estimate the coeffiscient.
        y : array
            The class for each epoch.

        Returns
        -------
        self : instance of CSP
            Returns the modified instance.
        
        References
        ----------
        """
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
        self.clf.fit(X, y)
        # computes the patterns
        if hasattr(self.clf, 'coef_'):
            self.patterns_ = np.dot(X.T, np.dot(X, self.clf.coef_.T))
            self.filters_ = self.clf.coef_
        
        return self
    
    
    def predict(self, X):
        """Predict class labels for each trial in X.
        
        Parameters
        ----------
        X : array, shape=(n_trials, n_features)
            The features of each trial to predict class label.
        Returns
        -------
        C : array, shape = [n_trials]
            Predicted class label per trials.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)."
                             % type(X))
        # check for features dimension
        X = np.atleast_2d(X)
        if len(X.shape) != 2:
            raise ValueError("X dimension should be 2 (n_trials x n_features)"
                             " instead of ", X.shape)
        
        y_pred = self.clf.predict(X)
        return y_pred
    
        
        